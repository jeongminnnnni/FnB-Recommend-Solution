"""
score_stability_consistency.py — SFT의 '출력 안정성'·'메뉴 일관성' 정량 지표 채점기

기존 평가 파이프라인(eval_baselines.py)이 저장한 raw 출력 로그(eval_raw.jsonl)를
재채점해, 기존 3대 지표에 빠져 있던 안정성/일관성 지표를 추가 계산한다.

[원칙]
  · 새 파서를 만들지 않는다. eval_baselines.extract_output / parse_int_won 을 그대로 재사용.
  · 입력의 보유재고·트렌드 어휘는 seed_data(PRICE_TABLE/TRENDS)에서 자동 구성.
  · 모델 출력 raw 텍스트만 입력으로 쓰고, GT/판정 로직은 건드리지 않는다(감사 가능).

[지표 정의]  ─ 사용자 확정안 반영 ─
  A1. cost 비정상 출력 비율(%)  : lenient 파싱 후 menu/calculation/cost 키 누락,
       또는 cost 파싱 불가/공백/<=0 인 출력의 비율. 분모=N. (낮을수록 안정)
  A2. 원가 추정 변동성          : 파싱된 model_cost 의 평균/표준편차(참고용).
  C1. 입력 재료 정합률(%)       : [토큰 정밀도] 메뉴에 등장한 '인식된 어휘'(재료/트렌드)
       중, 그 입력의 보유재고·트렌드(+트렌드 구성재료)에 실제 존재하는 비율.
       micro-average(어휘 등장 횟수 단위). (높을수록 일관/환각 적음)
  C2. 메뉴 중복률(%)            : [상황 단위] 20개 상황별 대표 메뉴를 뽑아, 서로 '다른
       상황' 간 정규화 일치 또는 부분문자열 관계로 충돌하는 상황의 비율.
       (같은 상황의 4개 가격이 같은 메뉴인 건 정상 → 제외). (낮을수록 입력 반응적)
"""

import sys
import types
import json
import re
import statistics
from collections import defaultdict, Counter

# ── eval_baselines 는 상단에서 torch 를 import 하므로, 채점에 불필요한 torch 를
#    stub 으로 주입해 파싱 함수만 재사용한다(모델 로드/추론 코드는 호출하지 않음). ──
if "torch" not in sys.modules:
    _t = types.ModuleType("torch")
    _t.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    _t.manual_seed = lambda *a, **k: None
    sys.modules["torch"] = _t

import eval_baselines as EB           # extract_output, parse_int_won 재사용
from seed_data import PRICE_TABLE, TRENDS

RAW_PATH = "eval_raw.jsonl"
REQUIRED_KEYS_A1 = ("menu", "calculation")   # + calculation.cost 별도 확인


# ─────────────────────────────────────────────────────────────
# 어휘/정규화 헬퍼
# ─────────────────────────────────────────────────────────────
def normalize(text):
    """공백·문장부호 제거 + 소문자화(부분문자열 매칭용)."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"[\s·()\[\]{}!@#$%^&*\-_,.~]", "", text).lower()


# 전역 인식 어휘집: 재료명 ∪ 트렌드명 ∪ 트렌드 구성재료 (정규화형 → 원형)
def build_vocab():
    vocab = {}
    for ing in PRICE_TABLE:
        vocab.setdefault(normalize(ing), ing)
    for tr in TRENDS:
        vocab.setdefault(normalize(tr["name"]), tr["name"])
        for ing in tr["ingredients"]:
            vocab.setdefault(normalize(ing), ing)
    # 너무 짧은 토큰(1글자)은 오매칭 위험 → 제외
    return {k: v for k, v in vocab.items() if len(k) >= 2}


VOCAB = build_vocab()
TREND_BY_NAME = {tr["name"]: tr for tr in TRENDS}


def parse_situation(situation):
    """'보유 재고: 크루아상·버터, 트렌드: 두바이 초콜릿' → (보유재고 리스트, 트렌드명)."""
    stock_part, _, trend_part = situation.partition(", 트렌드:")
    stock_part = stock_part.replace("보유 재고:", "").strip()
    trend = trend_part.strip()
    if stock_part in ("", "없음"):
        stocks = []
    else:
        stocks = [s.strip() for s in stock_part.split("·") if s.strip()]
    return stocks, trend


def allowed_terms_for(situation):
    """해당 입력에서 '정합'으로 인정되는 어휘 집합(정규화형)."""
    stocks, trend = parse_situation(situation)
    allowed = set()
    for s in stocks:
        allowed.add(normalize(s))
    # 트렌드명 자체 + 트렌드 구성재료
    allowed.add(normalize(trend))
    tr = TREND_BY_NAME.get(trend)
    if tr:
        for ing in tr["ingredients"]:
            allowed.add(normalize(ing))
    return {a for a in allowed if len(a) >= 2}


def recognized_terms(menu):
    """메뉴 문자열에 부분문자열로 등장하는 인식 어휘(정규화형) 집합."""
    nm = normalize(menu)
    if not nm:
        return set()
    return {key for key in VOCAB if key in nm}


# ─────────────────────────────────────────────────────────────
# 로드
# ─────────────────────────────────────────────────────────────
def load_rows():
    rows = defaultdict(list)
    with open(RAW_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["system"]].append(r)
    return rows


# ─────────────────────────────────────────────────────────────
# 지표 계산
# ─────────────────────────────────────────────────────────────
def score_system(records):
    n = len(records)

    # A1 / A2
    abnormal = 0
    costs = []
    for r in records:
        _, lenient = EB.extract_output(r["raw"])
        ok = isinstance(lenient, dict) and all(k in lenient for k in REQUIRED_KEYS_A1)
        cost_val = None
        if ok:
            calc = lenient.get("calculation")
            if isinstance(calc, dict) and "cost" in calc:
                cost_val = EB.parse_int_won(calc.get("cost"))
            else:
                ok = False
        if (not ok) or (cost_val is None) or (cost_val <= 0):
            abnormal += 1
        if cost_val is not None and cost_val > 0:
            costs.append(cost_val)

    # C1 (토큰 정밀도, micro-average): grounded 등장수 / 인식 등장수
    total_recognized = 0
    total_grounded = 0
    menus_with_no_recognized = 0
    for r in records:
        menu = r.get("menu")
        rec = recognized_terms(menu)
        if not rec:
            if menu and str(menu).strip():
                menus_with_no_recognized += 1
            continue
        allowed = allowed_terms_for(r["instruction"].split(", 판매가")[0])
        total_recognized += len(rec)
        total_grounded += len(rec & allowed)

    # C2 (상황 단위): 상황별 대표 메뉴 → 다른 상황 간 충돌 비율
    by_sit = defaultdict(list)
    for r in records:
        sit = r["instruction"].split(", 판매가")[0]
        by_sit[sit].append(r.get("menu"))
    rep = {}  # situation -> normalized representative menu
    for sit, menus in by_sit.items():
        norm_menus = [normalize(m) for m in menus if m and str(m).strip()]
        if not norm_menus:
            continue
        rep[sit] = Counter(norm_menus).most_common(1)[0][0]
    sits = list(rep.items())
    collided = set()
    for i in range(len(sits)):
        for j in range(i + 1, len(sits)):
            a, b = sits[i][1], sits[j][1]
            if not a or not b:
                continue
            if a == b or a in b or b in a:
                collided.add(sits[i][0])
                collided.add(sits[j][0])
    n_sit = len(rep)

    pct = lambda a, b: round(a / b * 100, 1) if b else 0.0
    return {
        "n": n,
        "A1_cost_abnormal_%": pct(abnormal, n),
        "A1_abnormal_count": abnormal,
        "A2_cost_mean": round(statistics.mean(costs), 1) if costs else None,
        "A2_cost_std": round(statistics.pstdev(costs), 1) if len(costs) > 1 else None,
        "A2_cost_n": len(costs),
        "C1_ingredient_match_%": pct(total_grounded, total_recognized),
        "C1_grounded": total_grounded,
        "C1_recognized": total_recognized,
        "C1_menus_no_vocab": menus_with_no_recognized,
        "C2_menu_dup_%": pct(len(collided), n_sit),
        "C2_collided_sit": len(collided),
        "C2_n_sit": n_sit,
    }


def main():
    rows = load_rows()
    label = {"b0": "B0 (zero-shot)", "b1": "B1 (few-shot)", "proposed": "Proposed (SFT+GR)"}
    results = {}
    for system in ("b0", "b1", "proposed"):
        results[system] = score_system(rows[system])

    # 콘솔 표
    print("=" * 78)
    print("안정성 / 일관성 지표 (raw 로그 재채점)")
    print("=" * 78)
    cols = [
        ("A1_cost_abnormal_%", "A1 비정상cost%↓", 16),
        ("C1_ingredient_match_%", "C1 재료정합%↑", 15),
        ("C2_menu_dup_%", "C2 메뉴중복%↓", 15),
        ("A2_cost_mean", "A2 평균원가", 12),
        ("A2_cost_std", "A2 표준편차", 12),
    ]
    header = f"{'system':<20}" + "".join(f"{d:<{w}}" for _, d, w in cols)
    print(header)
    print("-" * len(header))
    for system in ("b0", "b1", "proposed"):
        s = results[system]
        line = f"{label[system]:<20}"
        for key, _, w in cols:
            line += f"{str(s[key]):<{w}}"
        print(line)
    print("-" * len(header))
    for system in ("b0", "b1", "proposed"):
        s = results[system]
        print(f"  · {label[system]}: "
              f"A1 {s['A1_abnormal_count']}/{s['n']}  "
              f"C1 grounded {s['C1_grounded']}/{s['C1_recognized']} "
              f"(어휘 미인식 메뉴 {s['C1_menus_no_vocab']})  "
              f"C2 {s['C2_collided_sit']}/{s['C2_n_sit']} 상황 충돌")

    with open("eval_stability_consistency.json", "w", encoding="utf-8") as f:
        json.dump({label[k]: v for k, v in results.items()}, f, ensure_ascii=False, indent=2)
    print("\n✅ 저장: eval_stability_consistency.json")


if __name__ == "__main__":
    main()
