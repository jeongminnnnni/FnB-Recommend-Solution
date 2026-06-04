"""cot_dataset.jsonl의 calculation 필드 숫자 검증 스크립트.

검증 전략:
  1. (우선) seed_data.py 또는 generate_prompts.py 에서 레시피·단가표를 로드해
     margin_engine.evaluate()를 재실행하고 결과를 비교한다.
  2. (fallback) 해당 파일이 없으면, JSONL 안에 이미 적힌 cost·price 숫자를
     파싱한 뒤 margin_engine.calculate_margin()으로 마진율을 재계산하고
     적힌 마진율·status와 비교한다.

비교 항목: cost, margin(소수점 1자리), status(PASS/FAIL)
"""

import importlib
import json
import re
import sys
from pathlib import Path

from margin_engine import calculate_margin, evaluate

JSONL_PATH = Path("cot_dataset.jsonl")
SEED_CANDIDATES = ["seed_data", "generate_prompts"]
TARGET_MARGIN = 70.0


# ── 파서 헬퍼 ─────────────────────────────────────────────────────────────────

def parse_cost(text: str) -> int:
    """'776원' 또는 '1,925원' 등에서 정수 원가를 추출한다."""
    digits = re.sub(r"[,\s]", "", text)
    m = re.search(r"(\d+)", digits)
    if not m:
        raise ValueError(f"cost 파싱 실패: {text!r}")
    return int(m.group(1))


def parse_price_and_margin(text: str) -> tuple[int, float]:
    """'6000원 (마진율 87.1%)' 에서 (price, margin) 을 추출한다."""
    digits = re.sub(r"[,\s]", "", text)
    price_m = re.search(r"(\d+)원", digits)
    margin_m = re.search(r"(\d+(?:\.\d+)?)%", digits)
    if not price_m:
        raise ValueError(f"price 파싱 실패: {text!r}")
    if not margin_m:
        raise ValueError(f"margin 파싱 실패: {text!r}")
    return int(price_m.group(1)), float(margin_m.group(1))


# ── JSONL 로드 ────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[오류] {lineno}번째 줄 JSON 파싱 실패: {e}", file=sys.stderr)
    return entries


# ── seed 파일 탐지 ─────────────────────────────────────────────────────────────

def try_load_seed() -> tuple[dict | None, str | None]:
    """SEED_CANDIDATES 모듈에서 RECIPES와 PRICE_TABLE을 로드한다.
    성공하면 ({"recipes": ..., "price_table": ...}, module_name),
    실패하면 (None, None)을 반환한다.
    """
    for mod_name in SEED_CANDIDATES:
        try:
            mod = importlib.import_module(mod_name)
            recipes = getattr(mod, "RECIPES", None)
            price_table = getattr(mod, "PRICE_TABLE", None)
            if recipes and price_table:
                return {"recipes": recipes, "price_table": price_table}, mod_name
        except ModuleNotFoundError:
            continue
    return None, None


# ── 검증 로직 ─────────────────────────────────────────────────────────────────

def validate_with_seed(entries: list[dict], seed: dict) -> list[dict]:
    """seed 레시피를 이용해 evaluate()를 재실행하고 불일치를 반환한다."""
    recipe_map: dict[str, dict] = {r["menu"]: r for r in seed["recipes"]}
    price_table = seed["price_table"]
    mismatches = []

    for idx, entry in enumerate(entries, 1):
        menu = entry.get("menu", "")
        calc = entry.get("calculation", {})
        stated_cost = parse_cost(calc.get("cost", ""))
        _, stated_margin = parse_price_and_margin(calc.get("price", ""))
        stated_status = calc.get("status", "")

        if menu not in recipe_map:
            mismatches.append({
                "idx": idx, "menu": menu,
                "issue": "seed에서 메뉴를 찾을 수 없음",
                "stated": "", "computed": "",
            })
            continue

        rec = recipe_map[menu]
        result = evaluate(rec["recipe"], price_table, rec["price"])

        issues = []
        if result["cost"] != stated_cost:
            issues.append(f"cost: 기대 {result['cost']} / 기재 {stated_cost}")
        if result["margin"] != stated_margin:
            issues.append(f"margin: 기대 {result['margin']}% / 기재 {stated_margin}%")
        if result["status"] != stated_status:
            issues.append(f"status: 기대 {result['status']} / 기재 {stated_status}")

        if issues:
            mismatches.append({
                "idx": idx, "menu": menu,
                "issue": " | ".join(issues),
                "stated": f"cost={stated_cost}, margin={stated_margin}%, status={stated_status}",
                "computed": f"cost={result['cost']}, margin={result['margin']}%, status={result['status']}",
            })

    return mismatches


def validate_from_jsonl(entries: list[dict]) -> list[dict]:
    """JSONL에 적힌 cost·price로 margin을 재계산해 내부 정합성을 검증한다."""
    mismatches = []

    for idx, entry in enumerate(entries, 1):
        menu = entry.get("menu", "")
        calc = entry.get("calculation", {})

        stated_cost = parse_cost(calc.get("cost", ""))
        stated_price, stated_margin = parse_price_and_margin(calc.get("price", ""))
        stated_status = calc.get("status", "")

        computed_margin = calculate_margin(stated_cost, stated_price)
        computed_status = "PASS" if computed_margin >= TARGET_MARGIN else "FAIL"

        issues = []
        if computed_margin != stated_margin:
            issues.append(
                f"margin: 재계산 {computed_margin}% / 기재 {stated_margin}%"
            )
        if computed_status != stated_status:
            issues.append(
                f"status: 재계산 {computed_status} / 기재 {stated_status}"
            )

        if issues:
            mismatches.append({
                "idx": idx, "menu": menu,
                "issue": " | ".join(issues),
                "stated": f"cost={stated_cost}, price={stated_price}, margin={stated_margin}%, status={stated_status}",
                "computed": f"margin={computed_margin}%, status={computed_status}",
            })

    return mismatches


# ── 출력 ──────────────────────────────────────────────────────────────────────

def print_mismatches(mismatches: list[dict], total: int) -> None:
    if not mismatches:
        print(f"[OK] {total}개 전부 숫자 일치")
        return

    col_w = [5, 30, 42, 42]
    header = f"{'No':>{col_w[0]}}  {'메뉴명':<{col_w[1]}}  {'기재값':<{col_w[2]}}  {'재계산값'}"
    sep = "-" * (sum(col_w) + 6)
    print(f"\n불일치 항목 {len(mismatches)}개 / 전체 {total}개\n")
    print(header)
    print(sep)
    for m in mismatches:
        print(
            f"{m['idx']:>{col_w[0]}}  {m['menu']:<{col_w[1]}}  "
            f"{m['stated']:<{col_w[2]}}  {m['computed']}"
        )
        if m.get("issue"):
            print(f"{'':>{col_w[0]}}  └─ 불일치: {m['issue']}")
    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not JSONL_PATH.exists():
        sys.exit(f"[오류] {JSONL_PATH} 파일을 찾을 수 없습니다.")

    entries = load_jsonl(JSONL_PATH)
    if not entries:
        sys.exit("[오류] JSONL 파일이 비어 있습니다.")

    seed, seed_name = try_load_seed()

    if seed:
        print(f"[seed 모드] {seed_name}.py 로드 성공 → evaluate() 재실행으로 검증")
        mismatches = validate_with_seed(entries, seed)
    else:
        print("[fallback 모드] seed 파일 없음 → JSONL의 cost·price로 margin 재계산 검증")
        mismatches = validate_from_jsonl(entries)

    print_mismatches(mismatches, len(entries))


if __name__ == "__main__":
    main()
