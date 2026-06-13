"""
infer_test.py — 학습된 LoRA 어댑터 추론 검증 스크립트
프로젝트: FnB-Recommend-Solution (LLM 생성 + Rule-based Guardrail 하이브리드)

목적:
    train_sft.py 로 학습한 qwen3_4b_fnb_lora/final_adapter 어댑터가
    실제로 우리가 원하는 출력을 내는지 추론으로 확인한다.

확인 포인트(3가지):
    1) 출력이 '순수 JSON'으로 파싱되는가 (```json 백틱이나 잡설 없이)
    2) JSON 안에 reasoning / menu / calculation 키가 있는가
    3) 학습에 없던 새 입력(보유재고+트렌드 조합)에도 일반화되는가

설계 원칙:
    - 학습(train_sft.py)과 '완전히 동일한' 모델 로드 방식·chat template 사용.
      · device_map={"":0}, load_in_4bit=True, 사전양자화 4bit 미러(base)
      · get_chat_template(tokenizer, "qwen3-instruct")
    - 어댑터의 adapter_config.base_model_name_or_path 가 학습 때의 MODEL_NAME과
      동일하므로, 어댑터 디렉터리를 그대로 FastModel.from_pretrained 에 넘기면
      Unsloth 가 base(4bit) + LoRA 를 함께 로드한다(가장 학습과 동일한 경로).
    - [주의] Python 3.14 + datasets 의 map/fingerprint(dill) 버그를 피하기 위해
      추론에서는 datasets 의 map/fingerprint 경로를 일절 쓰지 않는다.
      (테스트 입력은 jsonl 직접 읽기 + 순수 Python 리스트로만 처리)

실행:
    cd /mnt/c/Users/MSI/Documents/GitHub/FnB-Recommend-Solution
    source sft_env/bin/activate
    python infer_test.py
"""

import os
# ★ 학습 스크립트와 동일하게: torch import 이전에 메모리 단편화 한도 설정.
#   (WSL2 에서 expandable_segments 는 미지원이므로 max_split_size 만 사용)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import json
import torch
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

# ─────────────────────────────────────────────────────────────
# 0. 설정 (train_sft.py 와 동일 값 + 추론 전용 파라미터)
# ─────────────────────────────────────────────────────────────
MODEL_NAME     = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"  # 학습 base (참고용)
ADAPTER_DIR    = "qwen3_4b_fnb_lora/final_adapter"  # 학습 산출물(LoRA). base 정보 내장
MAX_SEQ_LENGTH = 512        # 학습과 동일
DATA_PATH      = "sft_dataset.jsonl"

# 추론 파라미터: 재현성을 위해 그리디 디코딩(do_sample=False)
MAX_NEW_TOKENS = 512        # reasoning 최장 ~180토큰 + 여유
SEED           = 42

REQUIRED_KEYS  = ["reasoning", "menu", "calculation"]  # 검증할 최상위 키

# ─────────────────────────────────────────────────────────────
# 1. 테스트 입력 구성
#    (a) 학습 데이터에서 2개  → in-distribution (정상 재현 확인)
#    (b) 학습에 없는 새 조합 2개 → out-of-distribution (일반화 확인)
#
#    입력 포맷(학습과 동일):
#      "보유 재고: <재고>, 트렌드: <트렌드명>, 판매가 <N>원, 목표 마진: 70% 이상"
# ─────────────────────────────────────────────────────────────

# (a) 학습 데이터에서 2개를 '파일에서 직접' 읽어 정확히 동일한 입력 사용
#     (하드코딩 시 오타/공백 차이로 in-distribution 이 깨지는 것을 방지)
def load_indist_instructions(path, indices):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return [records[i]["instruction"] for i in indices]

INDIST_INDICES = [0, 24]  # 0: 두바이초콜릿/FAIL 사례, 24: 다른 트렌드 사례
indist_instructions = load_indist_instructions(DATA_PATH, INDIST_INDICES)

# (b) 학습에 없는 '새 조합' 2개 (seed_data.py 의 PRICE_TABLE/TRENDS/RECIPES 참고).
#     학습 데이터의 정규 조합에 '추가 보유 재고'를 끼워 넣어 명백히 OOD 로 만든다.
#     - new#1: 학습은 "보유 재고: 그릭요거트" 였음 → 여기선 "그릭요거트·딸기"(딸기 추가 보유)
#              딸기를 보유로 돌리면 신규 발주 원가가 줄어 마진 계산이 달라져야 함.
#     - new#2: 학습은 "보유 재고: 우유·생크림" 였음 → 여기선 "우유·생크림·말차파우더"
#              비싼 말차파우더(500g 30,000원)를 보유로 돌리는 새 상황.
new_combos = [
    "보유 재고: 그릭요거트·딸기, 트렌드: 그릭요거트볼, 판매가 6500원, 목표 마진: 70% 이상",
    "보유 재고: 우유·생크림·말차파우더, 트렌드: 말차 디저트, 판매가 6000원, 목표 마진: 70% 이상",
]

# (라벨, 입력, 유형) 튜플 리스트
test_cases = []
for i, ins in enumerate(indist_instructions, start=1):
    test_cases.append((f"in-dist #{i}", ins, "in-dist"))
for i, ins in enumerate(new_combos, start=1):
    test_cases.append((f"new #{i}", ins, "new"))

# ─────────────────────────────────────────────────────────────
# 2. 모델 로드 (학습과 동일한 방식)
#    어댑터 디렉터리를 그대로 넘기면 Unsloth 가 base(4bit) + LoRA 로드.
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("[1/3] 모델 + 어댑터 로드 중...")
print(f"      adapter: {ADAPTER_DIR}")
print("=" * 60)

model, tokenizer = FastModel.from_pretrained(
    model_name      = ADAPTER_DIR,   # base 정보는 adapter_config 에 내장됨
    max_seq_length  = MAX_SEQ_LENGTH,
    load_in_4bit    = True,          # 학습과 동일 (QLoRA 4bit base)
    full_finetuning = False,
    device_map      = {"": 0},       # 학습과 동일 (전부 0번 CUDA)
)

# 학습과 '완전히 동일한' chat template 적용
tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")

# 추론 최적화 모드 (Unsloth 2x faster inference)
FastModel.for_inference(model)
model.eval()

# pad 토큰 미설정 대비
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

# ─────────────────────────────────────────────────────────────
# 3. 생성 + 검증
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("[2/3] 추론 실행...")
print("=" * 60)

torch.manual_seed(SEED)


def generate(instruction: str) -> str:
    """단일 instruction → 모델 생성 텍스트(어시스턴트 turn 만) 반환."""
    messages = [{"role": "user", "content": instruction}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize               = True,
        add_generation_prompt  = True,   # 어시스턴트 응답을 생성하도록
        return_tensors         = "pt",
    ).to(model.device)

    with torch.inference_mode():
        out = model.generate(
            input_ids          = inputs,
            max_new_tokens     = MAX_NEW_TOKENS,
            do_sample          = False,   # 그리디(재현성)
            temperature        = None,
            top_p              = None,
            top_k              = None,
            pad_token_id       = tokenizer.pad_token_id,
        )
    # 프롬프트 부분을 잘라내고 새로 생성된 토큰만 디코딩
    gen_ids = out[0][inputs.shape[1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


def check(raw: str) -> dict:
    """생성 텍스트에 대해 3가지 확인 포인트를 자동 점검."""
    result = {
        "has_fence": ("```" in raw),     # 백틱 코드펜스 섞였는지(있으면 '순수 JSON' 위반)
        "json_ok": False,                # 점검①: 순수 JSON 파싱 성공 여부
        "keys_ok": False,                # 점검②: 필수 키 3종 모두 존재
        "missing": list(REQUIRED_KEYS),  # 빠진 키 목록
        "menu": None,
        "status": None,
    }
    try:
        obj = json.loads(raw)            # raw 그대로(스트립 후) 파싱 → '순수 JSON' 검증
        result["json_ok"] = True
        if isinstance(obj, dict):
            result["missing"] = [k for k in REQUIRED_KEYS if k not in obj]
            result["keys_ok"] = (len(result["missing"]) == 0)
            result["menu"] = obj.get("menu")
            calc = obj.get("calculation")
            if isinstance(calc, dict):
                result["status"] = calc.get("status")
    except json.JSONDecodeError:
        pass
    return result


rows = []
for label, instruction, kind in test_cases:
    print(f"\n[{label}] ({kind})")
    print(f"  입력: {instruction}")
    raw = generate(instruction)
    chk = check(raw)
    print(f"  출력(raw):\n    " + raw.replace("\n", "\n    "))
    rows.append((label, kind, instruction, raw, chk))

# ─────────────────────────────────────────────────────────────
# 4. 결과 표 출력
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("[3/3] 검증 결과 요약")
print("=" * 60)

def mark(b):  # bool → 기호
    return "PASS" if b else "FAIL"

header = f"{'case':<11} {'type':<8} {'json_ok':<8} {'no_fence':<9} {'keys_ok':<8} {'menu'}"
print(header)
print("-" * len(header))
all_pass = True
for label, kind, instruction, raw, chk in rows:
    json_ok  = chk["json_ok"]
    no_fence = (not chk["has_fence"])
    keys_ok  = chk["keys_ok"]
    case_pass = json_ok and no_fence and keys_ok
    all_pass &= case_pass
    menu = chk["menu"] if chk["menu"] is not None else "-"
    print(f"{label:<11} {kind:<8} {mark(json_ok):<8} {mark(no_fence):<9} "
          f"{mark(keys_ok):<8} {menu}")
    if not keys_ok and json_ok:
        print(f"            └ 빠진 키: {chk['missing']}")

print("-" * len(header))
print(f"\n종합: {'✅ 전체 통과' if all_pass else '⚠️  일부 실패 (위 표 확인)'}")
print("\n[해석 가이드]")
print("  · json_ok=PASS & no_fence=PASS  → 점검① '순수 JSON' 충족")
print("  · keys_ok=PASS                  → 점검② reasoning/menu/calculation 키 존재")
print("  · type=new 케이스가 통과         → 점검③ 미학습 조합 일반화 성공")
