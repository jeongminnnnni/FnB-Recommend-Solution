"""
train_sft.py — Gemma 4 E2B QLoRA SFT 학습 스크립트
프로젝트: FnB-Recommend-Solution (LLM 생성 + Rule-based Guardrail 하이브리드)

환경: Windows + WSL2, RTX 4060 Laptop 8GB, Unsloth 2026.6.1
실행:
    cd /mnt/c/Users/MSI/Documents/GitHub/FnB-Recommend-Solution
    source sft_env/bin/activate
    python train_sft.py

핵심 설계 의도(논문/발표 서술용):
- 8GB 소비자 GPU에서 재현 가능한 경량 셋업 (QLoRA 4bit + LoRA)
- 과적합 방지: 작은 r, epoch 2~3, eval loss 모니터링
- Gemma 4 E2B의 KV-share 이슈 회피: use_gradient_checkpointing="unsloth" 사용
  (일반 gradient_checkpointing=True 가 강제하는 use_cache=False 경로 대신
   Unsloth 전용 구현을 사용. 본 버전(2026.6.1)에는 관련 패치 포함됨)
"""

import os
import json
import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ─────────────────────────────────────────────────────────────
# 0. 설정 (한 곳에 모아서 논문 부록에 그대로 인용 가능)
# ─────────────────────────────────────────────────────────────
MODEL_NAME      = "unsloth/gemma-4-E2B-it"   # 게이트 동의 완료 가정. 안 되면 "google/gemma-4-e2b-it"
MAX_SEQ_LENGTH  = 1024     # 우리 데이터는 짧음(상황입력+JSON출력). 8GB 절약 위해 1024로 제한
DATA_PATH       = "sft_dataset.jsonl"
OUTPUT_DIR      = "gemma4_e2b_fnb_lora"
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
    # token = "hf_...",        # 환경변수 HF_TOKEN 사용 권장. 필요시 주석 해제
)

# Gemma 4 chat template 적용
tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

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
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("[3/5] 데이터셋 로드 & 포맷팅...")
print("=" * 60)

raw = load_dataset("json", data_files=DATA_PATH, split="train")
print(f"  전체 샘플 수: {len(raw)}")

def formatting_func(example):
    """instruction/output 쌍을 Gemma chat 텍스트로 변환.
    output은 '순수 JSON 문자열'이므로 그대로 assistant turn에 넣는다."""
    messages = [
        {"role": "user",      "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

dataset = raw.map(formatting_func, remove_columns=raw.column_names)

# train / eval split (과적합 모니터링용)
split = dataset.train_test_split(test_size=EVAL_SIZE, seed=SEED, shuffle=True)
train_dataset = split["train"]
eval_dataset  = split["test"]
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
    processing_class = tokenizer,   # TRL 0.24+ : `tokenizer=` 는 제거됨 → `processing_class`
    train_dataset    = train_dataset,
    eval_dataset     = eval_dataset,
    args = SFTConfig(
        dataset_text_field          = "text",
        max_length                  = MAX_SEQ_LENGTH,   # TRL 0.24+ : `max_seq_length` → `max_length`로 개명
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