"""
train_sft.py — Qwen3-4B-Instruct QLoRA SFT 학습 스크립트
프로젝트: FnB-Recommend-Solution (LLM 생성 + Rule-based Guardrail 하이브리드)

환경: Windows + WSL2, RTX 4060 Laptop 8GB, Unsloth 2026.6.1
실행:
    cd /mnt/c/Users/MSI/Documents/GitHub/FnB-Recommend-Solution
    source sft_env/bin/activate
    python train_sft.py

핵심 설계 의도(논문/발표 서술용):
- 8GB 소비자 GPU에서 재현 가능한 경량 셋업 (QLoRA 4bit + LoRA)
- 과적합 방지: 작은 r, epoch 2~3, eval loss 모니터링
- 모델 선택 근거: 당초 Gemma 4 E2B(멀티모달)를 시도했으나 4bit 로드만으로
  ~7.6GB(언어모델 6.66 + lm_head 0.75 + 비전/오디오 타워 0.9)를 점유해
  8GB GPU에서 학습이 구조적으로 불가능했다. 순수 텍스트 태스크(상황→JSON)이므로
  텍스트 전용 Qwen3-4B-Instruct로 교체. 4bit 상주 ~3GB로 학습 헤드룸이 충분하고,
  vocab(~15만)이 Gemma(26만)의 절반이라 fused cross-entropy 메모리 스파이크도 작다.
- use_gradient_checkpointing="unsloth": use_cache=False 강제 경로 대신 Unsloth
  전용 구현을 사용해 VRAM을 더 절약(본 버전 2026.6.1 패치 포함)
"""

import os
# ★ 8GB GPU OOM 완화: 큰 블록 재사용 실패 시 메모리 단편화를 줄이도록 분할 한도를 둔다.
#   주의) expandable_segments:True 는 WSL2 CUDA 드라이버가 VMM API를 미지원하여
#   로딩 단계에서 "CUDA driver error: out of memory" 를 유발하므로 WSL에서는 쓰지 않는다.
#   (반드시 torch import 이전에 설정해야 적용됨)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import json
import random
import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

# ─────────────────────────────────────────────────────────────
# Python 3.14 호환 패치
# datasets 4.3.0의 datasets.utils._dill.Pickler._batch_setitems 가 옛 시그니처
# (self, items) 라서, 3.14 pickle 이 (items, obj) 로 호출하면
# "_batch_setitems() takes 2 positional arguments but 3 were given" 로 죽는다.
# → obj 인자를 받아서 부모 구현으로 그대로 흘려보내도록 런타임 패치.
#   (datasets 가 3.14 대응 버전으로 올라가면 이 블록은 제거 가능)
# ─────────────────────────────────────────────────────────────
import sys as _sys
if _sys.hexversion >= 0x30E00A1:  # Python 3.14.0a1 이상
    import dill as _dill
    from datasets.utils import _dill as _ds_dill

    def _batch_setitems_py314(self, items, obj=None):
        if self._legacy_no_dict_keys_sorting:
            return super(_ds_dill.Pickler, self)._batch_setitems(items, obj)
        # dict 키 순서를 무시(원본 로직 유지)
        try:
            items = sorted(items)
        except Exception:
            from datasets.fingerprint import Hasher
            items = sorted(items, key=lambda x: Hasher.hash(x[0]))
        _dill.Pickler._batch_setitems(self, items, obj)

    _ds_dill.Pickler._batch_setitems = _batch_setitems_py314

# ─────────────────────────────────────────────────────────────
# 0. 설정 (한 곳에 모아서 논문 부록에 그대로 인용 가능)
# ─────────────────────────────────────────────────────────────
MODEL_NAME      = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"  # 텍스트 전용 4bit. 8GB에 학습 헤드룸 충분
MAX_SEQ_LENGTH  = 512      # 실측 데이터 최대 ~180토큰. 512면 여유 충분
DATA_PATH       = "sft_dataset.jsonl"
OUTPUT_DIR      = "qwen3_4b_fnb_lora"
EVAL_SIZE       = 10       # 100개 중 10개를 eval로 분리 (train 90 / eval 10)
SEED            = 42

# LoRA 하이퍼파라미터 (과적합 방지: r 작게)
LORA_R          = 8
LORA_ALPHA      = 16       # 통상 alpha = 2*r
LORA_DROPOUT    = 0.0      # Unsloth 최적화상 0 권장

# 학습 하이퍼파라미터
NUM_EPOCHS      = 3        # early-stopping-like: best eval loss 체크포인트 사용
LEARNING_RATE   = 2e-4
BATCH_SIZE      = 1        # 8GB 안전. effective batch = BATCH_SIZE * GRAD_ACCUM
GRAD_ACCUM      = 8        # effective batch = 8
WARMUP_RATIO    = 0.05
WEIGHT_DECAY    = 0.01

# ─────────────────────────────────────────────────────────────
# 1. 모델 로드 (QLoRA 4bit)
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("[1/5] 모델 로드 중...")
print("=" * 60)

model, tokenizer = FastModel.from_pretrained(
    model_name      = MODEL_NAME,
    max_seq_length  = MAX_SEQ_LENGTH,
    load_in_4bit    = True,    # QLoRA
    full_finetuning = False,
    # ★ transformers 5.5.0의 device_map 자동산정이 빈 GPU인데도 일부를
    #   CPU로 잘못 배정하는 버그가 있어, 전부 0번 CUDA 디바이스에 올리도록 명시.
    device_map      = {"": 0},
    # token = "hf_...",        # 환경변수 HF_TOKEN 사용 권장. 필요시 주석 해제
)

# Qwen3-Instruct chat template 적용 (non-thinking instruct 변형)
tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")

# ─────────────────────────────────────────────────────────────
# 2. LoRA 어댑터 부착
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("[2/5] LoRA 어댑터 부착...")
print("=" * 60)

model = FastModel.get_peft_model(
    model,
    r                         = LORA_R,
    lora_alpha                = LORA_ALPHA,
    lora_dropout              = LORA_DROPOUT,
    bias                      = "none",
    # ★ 우리 태스크는 순수 텍스트(상황입력→JSON)라 vision/audio 레이어 불필요.
    #   끄면 학습 파라미터·VRAM이 줄어 8GB에 안정적으로 들어간다.
    finetune_vision_layers    = False,
    finetune_language_layers  = True,
    finetune_attention_modules = True,
    finetune_mlp_modules      = True,
    target_modules            = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    use_gradient_checkpointing = "unsloth",  # ★ 핵심: KV-share 안전 + VRAM 절약
    random_state              = SEED,
)

# ─────────────────────────────────────────────────────────────
# 3. 데이터셋 로드 & 포맷팅
#    원본: {"instruction": "<상황입력>", "output": "<순수 JSON 문자열>"}
#    → Gemma chat 형식의 단일 "text" 필드로 변환
#
#    [주의] Python 3.14 + datasets 조합은 load_dataset / .map /
#    .train_test_split 내부의 dill fingerprint 해싱에서 깨진다
#    (Pickler._batch_setitems 시그니처 변경 버그). 따라서 데이터 처리는
#    순수 Python으로 수행하고, 마지막에 Dataset.from_list로 한 번만 변환한다.
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("[3/5] 데이터셋 로드 & 포맷팅...")
print("=" * 60)

# 3-1. jsonl 직접 읽기
records = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))
print(f"  전체 샘플 수: {len(records)}")

# 3-2. Gemma chat 텍스트로 포맷팅 (순수 Python 루프)
def to_text(example):
    """instruction/output 쌍을 Gemma chat 텍스트로 변환.
    output은 '순수 JSON 문자열'이므로 그대로 assistant turn에 넣는다."""
    messages = [
        {"role": "user",      "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

texts = [{"text": to_text(r)} for r in records]

# 3-3. train / eval split (순수 Python, 재현성 위해 seed 고정)
rng = random.Random(SEED)
rng.shuffle(texts)
eval_rows  = texts[:EVAL_SIZE]
train_rows = texts[EVAL_SIZE:]

# 3-4. 마지막에 한 번만 Dataset으로 변환 (fingerprint 해싱 미발생 경로)
train_dataset = Dataset.from_list(train_rows)
eval_dataset  = Dataset.from_list(eval_rows)
print(f"  train: {len(train_dataset)}  /  eval: {len(eval_dataset)}")

# 포맷 확인용 1건 출력 (디버깅)
print("\n  --- 포맷 샘플(첫 200자) ---")
print("  " + train_dataset[0]["text"][:200].replace("\n", "\n  "))
print("  ---------------------------\n")

# ─────────────────────────────────────────────────────────────
# 4. Trainer 구성 & 학습
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("[4/5] 학습 시작...")
print("=" * 60)

trainer = SFTTrainer(
    model            = model,
    processing_class = tokenizer,
    train_dataset    = train_dataset,
    eval_dataset    = eval_dataset,
    args = SFTConfig(
        dataset_text_field          = "text",
        max_length                  = MAX_SEQ_LENGTH,
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        warmup_ratio                = WARMUP_RATIO,
        num_train_epochs            = NUM_EPOCHS,
        learning_rate               = LEARNING_RATE,
        weight_decay                = WEIGHT_DECAY,
        logging_steps               = 1,
        optim                       = "adamw_8bit",   # VRAM 절약
        lr_scheduler_type           = "linear",
        seed                        = SEED,
        output_dir                  = OUTPUT_DIR,
        # ── 과적합 모니터링: 매 epoch eval + best 체크포인트 보존 ──
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        save_total_limit            = 2,
        report_to                   = "none",
        bf16                        = True,   # Ampere 이상(4060 OK). 문제 시 fp16=True로 교체
    ),
)

trainer_stats = trainer.train()

# ─────────────────────────────────────────────────────────────
# 5. 어댑터 저장
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("[5/5] LoRA 어댑터 저장...")
print("=" * 60)

final_dir = os.path.join(OUTPUT_DIR, "final_adapter")
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

# 학습 메타 기록 (논문 재현성 부록용)
meta = {
    "model": MODEL_NAME,
    "lora_r": LORA_R, "lora_alpha": LORA_ALPHA,
    "epochs": NUM_EPOCHS, "lr": LEARNING_RATE,
    "effective_batch": BATCH_SIZE * GRAD_ACCUM,
    "max_seq_length": MAX_SEQ_LENGTH,
    "train_size": len(train_dataset), "eval_size": len(eval_dataset),
    "train_runtime_sec": trainer_stats.metrics.get("train_runtime"),
}
with open(os.path.join(OUTPUT_DIR, "train_meta.json"), "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)

print(f"\n✅ 완료. 어댑터: {final_dir}")
print(f"   메타: {os.path.join(OUTPUT_DIR, 'train_meta.json')}")
print("\n[참고] loss가 13~15를 크게 넘으면(100/300 등) 비정상 신호.")
print("       OOM 발생 시: MAX_SEQ_LENGTH를 768/512로, 또는 GRAD_ACCUM을 4로 낮추세요.")