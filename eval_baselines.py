"""
eval_baselines.py — B0 / B1 / 제안(SFT+Guardrail) 3단 비교 평가 파이프라인
프로젝트: FnB-Recommend-Solution (LLM 생성 + Rule-based Guardrail 하이브리드)

목적(연구 핵심):
    같은 held-out 입력셋에 세 시스템을 돌려, 3개 지표를 자동 측정·비교한다.
      · B0       : 순정 Qwen3-4B-Instruct-2507 (파인튜닝 X, zero-shot 프롬프트)
      · B1       : 순정 모델 + few-shot(K=3) 프롬프팅
      · Proposed : SFT 어댑터(final_adapter) + Rule Guardrail(margin_engine)
    B1을 끼워 "SFT의 순수 기여 vs Guardrail의 순수 기여"를 분리 측정한다.

평가 지표(3개, 전부 코드로 자동 측정):
    1) 마진 제약 위반율 : 시스템이 'green-light(배포)'한 추천 중, Guardrail이 목표
       마진(70%) 미달로 판정한 비율.  Proposed는 Guardrail로 미달을 100% 필터 → 0% 기대.
    2) 산술 정확도      : 모델이 self-report한 원가(cost)가 margin_engine GT와 일치하는 비율.
    3) JSON 포맷 준수율 : 출력이 '순수 JSON'으로 파싱되고 필수 키(reasoning/menu/calculation)를 갖춘 비율.

────────────────────────────────────────────────────────────────────────────
[핵심 설계: 방안 C — 입력 고정(input-anchored)]
모델 출력 JSON에는 recipe 필드가 없다(키: reasoning/menu/calculation). 따라서
"모델 menu → recipe" 변환을 모델 출력에 의존하지 않는다. 대신:
  · Guardrail evaluate() 의 입력 recipe·price 는 '평가 입력이 고정'한 canonical
    값(seed_data.RECIPES + 입력 판매가)을 쓴다. → 세 시스템이 '완전히 동일한
    recipe·price 기반'으로 채점되어 공정.
  · 모델 출력에서 평가로 가져오는 것은 오직:
      - 순수 JSON 파싱/필수키       → 지표3(포맷)
      - self-report cost/margin     → 지표2(산술정확도, GT=margin_engine)
      - menu 문자열                 → (보조) 'green-light 했는가' 판단/작명 진단
  · 최종 PASS/FAIL 은 '항상' margin_engine.evaluate() 가 독립적으로 내린다.
    모델의 self-report status 는 최종 판정에 쓰지 않는다(지표2 측정에만 사용).

[프롬프트 대칭성]
세 시스템 모두 입력의 '내용'(상황+판매가)은 동일. 단 각자 의도된 구성으로 실행:
  · B0  : system(태스크+스키마 설명) + user(상황)          — 예시 없음
  · B1  : system(동일) + K=3 데모(user/assistant 턴) + user — 데모는 '학습셋'에서
          추출(held-out 누설 없음), PASS/FAIL 섞음
  · Proposed : user(상황)만 — 스키마는 학습으로 가중치에 내재 + Guardrail
이로써 "스키마/원가 지식이 프롬프트에 있나 vs 가중치에 있나"만 달라져 SFT 기여를 격리.
────────────────────────────────────────────────────────────────────────────

[모델 로드] train_sft.py 와 동일: load_in_4bit=True, device_map={"":0}, 사전양자화 4bit 미러.
    · B0/B1   : base(MODEL_NAME) 를 어댑터 없이 로드 (둘은 같은 모델 → 한 번 로드해 공용)
    · Proposed: 어댑터 디렉터리 로드 (Unsloth 가 base+LoRA 동시 로드)
    8GB VRAM 고려: base 로 B0·B1 모두 처리 후 free → 어댑터 로드해 Proposed 처리.

[출력물]
    · 콘솔 요약표
    · eval_results.csv   — 시스템×지표 요약 (논문 결과표 직행)
    · eval_results.json  — 설정 + 테스트셋 요약 + 시스템별 전체 지표
    · eval_raw.jsonl     — 입력×시스템 raw 출력 + 추출값 + GT (재현/감사용)

[Py3.14 주의] datasets 의 map/load_dataset/fingerprint 경로를 일절 쓰지 않는다.
    (테스트셋은 seed_data + jsonl 직접 읽기 + 순수 Python 으로만 구성)

실행:
    cd /mnt/c/Users/MSI/Documents/GitHub/FnB-Recommend-Solution
    source sft_env/bin/activate
    python eval_baselines.py
"""

import os
# ★ train_sft.py 와 동일: torch import 이전에 메모리 단편화 한도 설정(WSL2 호환).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

import gc
import re
import csv
import json
import torch

from margin_engine import evaluate, calculate_cost
from seed_data import RECIPES, PRICE_TABLE

# ─────────────────────────────────────────────────────────────
# Python 3.14 호환 패치 (train_sft.py 와 동일).
#   본 스크립트는 datasets 의 map/fingerprint 를 쓰지 않으므로 원칙적으로 불필요하나,
#   unsloth/trl 가 내부적으로 datasets 를 건드릴 가능성에 대비해 방어적으로 둔다.
# ─────────────────────────────────────────────────────────────
import sys as _sys
if _sys.hexversion >= 0x30E00A1:  # Python 3.14.0a1 이상
    try:
        import dill as _dill
        from datasets.utils import _dill as _ds_dill

        def _batch_setitems_py314(self, items, obj=None):
            if self._legacy_no_dict_keys_sorting:
                return super(_ds_dill.Pickler, self)._batch_setitems(items, obj)
            try:
                items = sorted(items)
            except Exception:
                from datasets.fingerprint import Hasher
                items = sorted(items, key=lambda x: Hasher.hash(x[0]))
            _dill.Pickler._batch_setitems(self, items, obj)

        _ds_dill.Pickler._batch_setitems = _batch_setitems_py314
    except Exception:
        pass  # datasets 미존재/구조변경 시 무시(우리 경로는 datasets 미사용)

# ─────────────────────────────────────────────────────────────
# 0. 설정 (train_sft.py 와 동일 값 + 평가 전용 파라미터)
# ─────────────────────────────────────────────────────────────
MODEL_NAME     = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"  # base(B0/B1)
ADAPTER_DIR    = "qwen3_4b_fnb_lora/final_adapter"                   # Proposed(SFT)
MAX_SEQ_LENGTH = 512
SFT_DATA_PATH  = "sft_dataset.jsonl"   # 학습 입력(가격) 제외 + B1 데모 추출용
TARGET_MARGIN  = 70.0

# 추론(재현성: greedy)
MAX_NEW_TOKENS = 512
SEED           = 42

# 테스트셋(확정안): 20 레시피 × (FAIL 2 + PASS 2) = 80, 50:50
N_FAIL_PER_RECIPE = 2
N_PASS_PER_RECIPE = 2
PRICE_STEP        = 50      # 가격 후보 격자(50원). 학습가(대개 100원 격자)와 자연히 어긋남

# B1 few-shot
N_FEWSHOT = 3

REQUIRED_KEYS = ["reasoning", "menu", "calculation"]

# ─────────────────────────────────────────────────────────────
# 1. 파싱 헬퍼 (validate_numbers.py 의 정규식 규약 재사용)
# ─────────────────────────────────────────────────────────────
def parse_int_won(text):
    """'1,386원' / '1386 원' 등에서 정수만 추출. 실패 시 None."""
    if not isinstance(text, str):
        return None
    digits = re.sub(r"[,\s]", "", text)
    m = re.search(r"(\d+)", digits)
    return int(m.group(1)) if m else None


def parse_price_margin(text):
    """'4700원 (마진율 70.5%)' → (price, margin). 실패 항목은 None."""
    if not isinstance(text, str):
        return None, None
    digits = re.sub(r"[,\s]", "", text)
    pm = re.search(r"(\d+)원", digits)
    mm = re.search(r"(\d+(?:\.\d+)?)%", digits)
    price = int(pm.group(1)) if pm else None
    margin = float(mm.group(1)) if mm else None
    return price, margin


def extract_output(raw):
    """모델 raw 출력에서 (strict_obj, lenient_obj) 반환.
    strict_obj : raw 를 그대로 json.loads (순수 JSON 검증용; 실패 시 None)
    lenient_obj: 코드펜스/잡설을 벗겨 최외곽 {..} 만 파싱 (지표2 측정을 위해 관대하게)
    """
    s = raw.strip()
    strict_obj = None
    try:
        loaded = json.loads(s)
        if isinstance(loaded, dict):
            strict_obj = loaded
    except json.JSONDecodeError:
        pass

    lenient_obj = strict_obj
    if lenient_obj is None:
        t = re.sub(r"```[a-zA-Z]*", "", s).replace("```", "")
        i, j = t.find("{"), t.rfind("}")
        if i != -1 and j != -1 and j > i:
            try:
                loaded = json.loads(t[i:j + 1])
                if isinstance(loaded, dict):
                    lenient_obj = loaded
            except json.JSONDecodeError:
                pass
    return strict_obj, lenient_obj


def is_format_ok(strict_obj):
    """지표3: 순수 JSON(strict) + 필수 키 3종 보유."""
    if not isinstance(strict_obj, dict):
        return False
    return all(k in strict_obj for k in REQUIRED_KEYS)


# ─────────────────────────────────────────────────────────────
# 2. 테스트셋 빌드 (held-out: 학습에 쓴 가격 제외, PASS/FAIL 균형)
# ─────────────────────────────────────────────────────────────
def load_training_prices_per_situation(path):
    """sft_dataset.jsonl 에서 (상황 → 사용된 판매가 집합) 추출."""
    used = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ins = rec["instruction"]
            sit, _, rest = ins.partition(", 판매가 ")
            try:
                price = int(rest.split("원")[0])
            except (ValueError, IndexError):
                continue
            used.setdefault(sit, set()).add(price)
    return used


def breakeven_price(recipe):
    """목표 마진(70%) 을 처음으로 충족하는 최소 정수 판매가."""
    cost = calculate_cost(recipe, PRICE_TABLE)
    p = cost + 1
    while True:
        if evaluate(recipe, PRICE_TABLE, p)["status"] == "PASS":
            return p
        p += 1


def build_testset():
    """held-out 평가 입력셋 + GT 부착 리스트 반환."""
    used = load_training_prices_per_situation(SFT_DATA_PATH)
    rows = []
    for rec in RECIPES:
        recipe = rec["recipe"]
        sit = rec["situation"]
        be = breakeven_price(recipe)
        train_prices = used.get(sit, set())

        # 가격 후보 격자: PRICE_STEP 의 배수
        grid = list(range(PRICE_STEP, be + PRICE_STEP * 80, PRICE_STEP))
        # FAIL: break-even 미만 + 학습가 제외 → break-even 에 '가까운(높은)' 순으로
        fails = sorted(
            [p for p in grid if p < be and p not in train_prices
             and evaluate(recipe, PRICE_TABLE, p)["status"] == "FAIL"],
            reverse=True,
        )[:N_FAIL_PER_RECIPE]
        # PASS: break-even 이상 + 학습가 제외 → break-even 에 '가까운(낮은)' 순으로
        passes = sorted(
            [p for p in grid if p not in train_prices
             and evaluate(recipe, PRICE_TABLE, p)["status"] == "PASS"]
        )[:N_PASS_PER_RECIPE]

        for price in fails + passes:
            res = evaluate(recipe, PRICE_TABLE, price)
            instruction = f"{sit}, 판매가 {price}원, 목표 마진: 70% 이상"
            rows.append({
                "instruction": instruction,
                "situation":   sit,
                "canon_menu":  rec["menu"],
                "price":       price,
                # ── GT (margin_engine) ──
                "gt_cost":        res["cost"],
                "gt_margin":      res["margin"],
                "gt_status":      res["status"],
                "gt_stock_usage": res["stock_usage"],
            })
    return rows


# ─────────────────────────────────────────────────────────────
# 3. B1 few-shot 데모 (학습셋에서 PASS/FAIL 섞어 K개, 고정)
# ─────────────────────────────────────────────────────────────
def load_fewshot_demos(path, k):
    """학습셋에서 데모 k개 선택: 최소 1 PASS + 1 FAIL 포함, 결정적 순서."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    def status_of(rec):
        try:
            return json.loads(rec["output"]).get("calculation", {}).get("status")
        except json.JSONDecodeError:
            return None

    first_fail = next((r for r in records if status_of(r) == "FAIL"), None)
    first_pass = next((r for r in records if status_of(r) == "PASS"), None)
    chosen = [r for r in (first_fail, first_pass) if r is not None]
    # 나머지는 앞에서부터 채워 k개
    for r in records:
        if len(chosen) >= k:
            break
        if r not in chosen:
            chosen.append(r)
    return chosen[:k]


# ─────────────────────────────────────────────────────────────
# 4. 프롬프트 빌더
# ─────────────────────────────────────────────────────────────
TASK_SPEC = (
    "당신은 카페 신메뉴 추천·원가 분석 도우미입니다. 사용자가 (보유 재고, 트렌드, "
    "판매가, 목표 마진)을 주면, 트렌드에 맞는 신메뉴 1개를 제안하고 재료 원가와 "
    "마진을 계산하세요. 반드시 아래 스키마의 '순수 JSON'만 출력하세요. "
    "코드블록(```), 추가 설명 문장, 그 외 어떤 텍스트도 절대 덧붙이지 마세요.\n"
    '스키마: {"reasoning": "<선정 이유와 마진 판단>", "menu": "<메뉴명>", '
    '"calculation": {"cost": "<원가>원", "price": "<판매가>원 (마진율 <NN.N>%)", '
    '"status": "PASS 또는 FAIL"}}'
)


def make_b0_prompt(instruction):
    return [
        {"role": "system", "content": TASK_SPEC},
        {"role": "user",   "content": instruction},
    ]


def make_b1_prompt(instruction, demos):
    messages = [{"role": "system", "content": TASK_SPEC}]
    for d in demos:
        messages.append({"role": "user",      "content": d["instruction"]})
        messages.append({"role": "assistant", "content": d["output"]})
    messages.append({"role": "user", "content": instruction})
    return messages


def make_proposed_prompt(instruction):
    # 학습 때와 동일하게 '상황 문자열'만 user 턴으로
    return [{"role": "user", "content": instruction}]


# ─────────────────────────────────────────────────────────────
# 5. 모델 로드 / 생성 / 해제
# ─────────────────────────────────────────────────────────────
def load_model(model_name):
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template
    model, tokenizer = FastModel.from_pretrained(
        model_name      = model_name,
        max_seq_length  = MAX_SEQ_LENGTH,
        load_in_4bit    = True,
        full_finetuning = False,
        device_map      = {"": 0},
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")
    FastModel.for_inference(model)
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def free_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_one(model, tokenizer, messages):
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize              = True,
        add_generation_prompt = True,
        return_tensors        = "pt",
    ).to(model.device)
    with torch.inference_mode():
        out = model.generate(
            input_ids      = ids,
            max_new_tokens = MAX_NEW_TOKENS,
            do_sample      = False,
            temperature    = None,
            top_p          = None,
            top_k          = None,
            pad_token_id   = tokenizer.pad_token_id,
        )
    gen = out[0][ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def run_system(model, tokenizer, testset, prompt_builder, system_name):
    """한 시스템을 전체 입력셋에 실행 → 입력별 raw 출력 리스트 반환."""
    raws = []
    n = len(testset)
    for i, row in enumerate(testset, 1):
        raw = generate_one(model, tokenizer, prompt_builder(row["instruction"]))
        raws.append(raw)
        print(f"  [{system_name}] {i}/{n} 생성 완료", end="\r", flush=True)
    print()
    return raws


# ─────────────────────────────────────────────────────────────
# 6. 채점 (방안 C: GT 는 margin_engine, 모델은 menu/self-calc/포맷만)
# ─────────────────────────────────────────────────────────────
def grade_row(system, raw, gt):
    """단일 (시스템, 출력, GT) 채점 → 상세 dict."""
    strict_obj, lenient_obj = extract_output(raw)

    # 지표3: 포맷
    format_ok = is_format_ok(strict_obj)

    # 모델 self-report 추출(관대): cost / price / margin / menu
    menu = lenient_obj.get("menu") if isinstance(lenient_obj, dict) else None
    calc = lenient_obj.get("calculation") if isinstance(lenient_obj, dict) else None
    model_cost = model_price = model_margin = None
    if isinstance(calc, dict):
        model_cost = parse_int_won(calc.get("cost"))
        model_price, model_margin = parse_price_margin(calc.get("price"))

    # 지표2: 산술 정확도 (모델 cost == GT cost). price echo / margin 일치는 보조 진단.
    arithmetic_correct = (model_cost is not None and model_cost == gt["gt_cost"])
    margin_match = (model_margin is not None and abs(model_margin - gt["gt_margin"]) < 1e-9)
    price_echo_ok = (model_price is not None and model_price == gt["price"])

    # green-light 판정:
    #   · proposed : Guardrail(=GT) 가 PASS 일 때만 배포 → GT FAIL 은 자동 필터
    #   · b0/b1    : Guardrail 없음 → 메뉴를 추천했으면(파싱되면) 그대로 배포
    recommended = (menu is not None and str(menu).strip() != "")
    if system == "proposed":
        green_lit = (gt["gt_status"] == "PASS")
    else:  # b0, b1
        green_lit = recommended

    # 지표1: green-light 한 추천이 실제로 마진 미달(FAIL)이면 위반
    violation = bool(green_lit and gt["gt_status"] == "FAIL")

    return {
        "format_ok":          format_ok,
        "recommended":        recommended,
        "menu":               menu,
        "model_cost":         model_cost,
        "model_price":        model_price,
        "model_margin":       model_margin,
        "arithmetic_correct": arithmetic_correct,
        "margin_match":       margin_match,
        "price_echo_ok":      price_echo_ok,
        "green_lit":          green_lit,
        "violation":          violation,
    }


def aggregate(system, graded, testset):
    """시스템 단위 3대 지표 + 보조 지표 집계.

    [지표2 분해] LLM 은 정수 산술을 정확히 재현하지 못해(SFT 포함) exact 일치는 거의 0.
    이를 단일 비율로 뭉개면 'exact≈0 → Guardrail 필요'와 'SFT 가 가장 근접'이 둘 다
    가려진다. 따라서 산술 정확도를 3개 하위지표로 분해해 정직하게 보고한다:
      · arithmetic_exact_%      : model_cost == GT cost (엄격; Guardrail 필요성 근거)
      · arithmetic_within5pct_% : |오차|/GT <= 5%  (SFT '근접' 학습 효과가 드러나는 지표)
      · cost_median_pct_err     : 파싱된 추정의 절대오차%(중앙값) — 작을수록 좋음
    (분모: exact/within 은 전체 N=미파싱은 비정확으로 간주, median 은 파싱분만)
    """
    import statistics

    n = len(graded)
    n_fail = sum(1 for r in testset if r["gt_status"] == "FAIL")

    violations = sum(1 for g in graded if g["violation"])
    fmt_ok     = sum(1 for g in graded if g["format_ok"])
    covered    = sum(1 for g in graded if g["recommended"])

    exact = within5 = within10 = 0
    abs_errs, pct_errs = [], []
    for g, row in zip(graded, testset):
        mc = g["model_cost"]
        if mc is None:
            continue
        gt = row["gt_cost"]
        ae = abs(mc - gt)
        pe = (ae / gt * 100) if gt else 0.0
        abs_errs.append(ae)
        pct_errs.append(pe)
        if mc == gt:
            exact += 1
        if pe <= 5:
            within5 += 1
        if pe <= 10:
            within10 += 1
    parseable = len(abs_errs)

    pct = lambda a, b: round(a / b * 100, 1) if b else 0.0
    return {
        "system": system,
        "n": n,
        "n_fail_scenarios": n_fail,
        # ── 지표1 ──
        "margin_violation_rate_%": pct(violations, n_fail),   # FAIL 시나리오 분모
        # ── 지표2(분해) ──
        "arithmetic_exact_%":      pct(exact, n),
        "arithmetic_within5pct_%": pct(within5, n),
        "cost_median_pct_err":     round(statistics.median(pct_errs), 1) if pct_errs else None,
        # ── 지표3 ──
        "format_compliance_%":     pct(fmt_ok, n),
        # ── 보조 지표 ──
        "recommendation_coverage_%": pct(covered, n),         # 유효추천 비율(착시 방지)
        "arithmetic_within10pct_%":  pct(within10, n),
        "cost_MAE_won":              round(statistics.mean(abs_errs), 1) if abs_errs else None,
        "parseable_cost_count":      parseable,
        "green_lit_count":           sum(1 for g in graded if g["green_lit"]),
        "violation_count":           violations,
    }


# ─────────────────────────────────────────────────────────────
# 7. 메인
# ─────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)

    print("=" * 64)
    print("[1/4] held-out 테스트셋 구성")
    print("=" * 64)
    testset = build_testset()
    n_pass = sum(1 for r in testset if r["gt_status"] == "PASS")
    n_fail = sum(1 for r in testset if r["gt_status"] == "FAIL")
    print(f"  테스트셋 크기: {len(testset)}  (PASS {n_pass} / FAIL {n_fail})")
    demos = load_fewshot_demos(SFT_DATA_PATH, N_FEWSHOT)
    demo_status = [json.loads(d["output"]).get("calculation", {}).get("status") for d in demos]
    print(f"  B1 few-shot 데모 {len(demos)}개 (status: {demo_status})")

    # 시스템별 raw 출력 수집
    raws_by_system = {}

    print("\n" + "=" * 64)
    print("[2/4] base 모델 로드 → B0 / B1 추론")
    print("=" * 64)
    base_model, base_tok = load_model(MODEL_NAME)
    raws_by_system["b0"] = run_system(
        base_model, base_tok, testset, make_b0_prompt, "B0")
    raws_by_system["b1"] = run_system(
        base_model, base_tok, testset,
        lambda ins: make_b1_prompt(ins, demos), "B1")
    free_model(base_model, base_tok)

    print("\n" + "=" * 64)
    print("[3/4] 어댑터(SFT) 로드 → Proposed 추론")
    print("=" * 64)
    sft_model, sft_tok = load_model(ADAPTER_DIR)
    raws_by_system["proposed"] = run_system(
        sft_model, sft_tok, testset, make_proposed_prompt, "Proposed")
    free_model(sft_model, sft_tok)

    print("\n" + "=" * 64)
    print("[4/4] 채점 & 집계 (GT = margin_engine)")
    print("=" * 64)

    summaries = []
    raw_records = []
    for system in ("b0", "b1", "proposed"):
        graded = []
        for row, raw in zip(testset, raws_by_system[system]):
            g = grade_row(system, raw, row)
            graded.append(g)
            raw_records.append({
                "system":      system,
                "instruction": row["instruction"],
                "gt_cost":     row["gt_cost"],
                "gt_margin":   row["gt_margin"],
                "gt_status":   row["gt_status"],
                "raw":         raw,
                **{k: g[k] for k in (
                    "format_ok", "menu", "model_cost", "model_margin",
                    "arithmetic_correct", "green_lit", "violation")},
            })
        summaries.append(aggregate(system, graded, testset))

    # ── 콘솔 표 ──
    label = {"b0": "B0 (zero-shot)", "b1": "B1 (few-shot)", "proposed": "Proposed (SFT+GR)"}
    cols = [
        ("system", "system", 18),
        ("format_compliance_%", "포맷%", 8),
        ("arithmetic_exact_%", "exact%", 8),
        ("arithmetic_within5pct_%", "≤5%근접", 9),
        ("cost_median_pct_err", "중앙오차%", 10),
        ("margin_violation_rate_%", "마진위반%", 10),
        ("recommendation_coverage_%", "추천커버%", 10),
    ]
    header = "".join(f"{disp:<{w}}" for _, disp, w in cols)
    print("\n" + header)
    print("-" * len(header))
    for s in summaries:
        line = ""
        for key, _, w in cols:
            val = label[s["system"]] if key == "system" else s[key]
            line += f"{str(val):<{w}}"
        print(line)
    print("-" * len(header))
    print("  · 지표1 마진위반%: 분모 = FAIL 시나리오 수 / Proposed 는 Guardrail 필터 → 0% 기대")
    print("  · 지표2(산술): exact%(정확 일치) + ≤5%근접%(SFT 근접 학습) + 중앙오차%(작을수록↑)")
    print("    └ exact≈0 = 'LLM 산술 비신뢰 → 결정론적 Guardrail 필요'의 직접 근거")
    print("  · 지표3 포맷%: 순수 JSON + 필수키(reasoning/menu/calculation)")
    print("  · 추천커버%: 유효추천(파싱+menu) 비율 — baseline '무응답=안전' 착시 방지용")

    # ── 저장: CSV ──
    csv_path = "eval_results.csv"
    fieldnames = [
        "system", "n", "n_fail_scenarios",
        "format_compliance_%",
        "arithmetic_exact_%", "arithmetic_within5pct_%", "cost_median_pct_err",
        "margin_violation_rate_%", "recommendation_coverage_%",
        "arithmetic_within10pct_%", "cost_MAE_won", "parseable_cost_count",
        "green_lit_count", "violation_count",
    ]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s[k] for k in fieldnames})

    # ── 저장: JSON(설정+테스트셋요약+시스템지표) ──
    json_path = "eval_results.json"
    payload = {
        "config": {
            "base_model": MODEL_NAME,
            "adapter": ADAPTER_DIR,
            "target_margin": TARGET_MARGIN,
            "decoding": {"do_sample": False, "max_new_tokens": MAX_NEW_TOKENS, "seed": SEED},
            "mapping_method": "C-input-anchored (recipe·price fixed by input; GT=margin_engine)",
            "fewshot_k": N_FEWSHOT,
        },
        "testset": {"size": len(testset), "pass": n_pass, "fail": n_fail},
        "results": summaries,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ── 저장: 입력×시스템 raw (재현/감사) ──
    raw_path = "eval_raw.jsonl"
    with open(raw_path, "w", encoding="utf-8") as f:
        for rec in raw_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ 저장: {csv_path} / {json_path} / {raw_path}")


if __name__ == "__main__":
    main()
