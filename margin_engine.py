"""Guardrail 계산 엔진: LLM 없이 순수 Python으로 동작하는 결정론적 원가/마진 계산기.

이 프로젝트의 마진 보장을 책임지는 핵심 모듈로, 외부 라이브러리 없이 표준 라이브러리만 사용한다.

도메인 규칙:
- 마진율 = (price - cost) / price * 100  (%)
- 원가(cost)는 "재료비만" 합산한다. 인건비/임대료/운영비는 제외.
- 재료비 = 각 재료의 (사용량 / 단위량) * 단위단가 의 총합
- 보유재고 활용률 = (보유재고로 충당되는 재료비 / 전체 재료비) * 100
- 목표 마진 기본값은 70%이며, 실제 마진 >= 목표 마진 이면 PASS.
"""

from typing import TypedDict

# 입력 데이터 구조에 대한 타입 별칭
PriceTable = dict[str, dict[str, float]]
Recipe = list[dict[str, object]]


class UnknownIngredientError(KeyError):
    """단가표에 없는 재료가 레시피에 포함되어 있을 때 발생하는 예외."""


class EvaluationResult(TypedDict):
    """evaluate()가 반환하는 판정 결과 구조."""

    cost: int
    price: int
    margin: float
    stock_usage: float
    target_margin: float
    status: str


def _ingredient_cost(item: dict[str, object], price_table: PriceTable) -> float:
    """레시피 항목 하나의 재료비를 (사용량 / 단위량) * 단위단가 로 계산한다(반올림 전 원시값)."""
    name = str(item["name"])
    if name not in price_table:
        raise UnknownIngredientError(
            f"단가표에 없는 재료입니다: '{name}'. 단가표에 unit_amount/unit_price를 먼저 등록하세요."
        )
    spec = price_table[name]
    amount = float(item["amount"])  # type: ignore[arg-type]
    return amount / spec["unit_amount"] * spec["unit_price"]


def calculate_cost(recipe: Recipe, price_table: PriceTable) -> int:
    """레시피의 총 재료원가(원)를 반올림한 정수로 반환한다."""
    total = sum(_ingredient_cost(item, price_table) for item in recipe)
    return round(total)


def calculate_margin(cost: float, price: float) -> float:
    """원가와 판매가로부터 마진율(%)을 소수점 1자리까지 계산한다."""
    if price <= 0:
        raise ValueError(f"판매가는 0보다 커야 합니다: {price}")
    return round((price - cost) / price * 100, 1)


def calculate_stock_usage(recipe: Recipe, price_table: PriceTable) -> float:
    """전체 재료비 대비 보유재고(from_stock=True)로 충당되는 재료비의 비율(%)을 소수점 1자리까지 반환한다."""
    total = 0.0
    stock = 0.0
    for item in recipe:
        cost = _ingredient_cost(item, price_table)
        total += cost
        if item.get("from_stock"):
            stock += cost
    if total == 0:
        return 0.0
    return round(stock / total * 100, 1)


def evaluate(
    recipe: Recipe,
    price_table: PriceTable,
    price: float,
    target_margin: float = 70,
) -> EvaluationResult:
    """레시피와 판매가를 평가해 원가/마진/재고활용률/목표/판정(PASS|FAIL)을 담은 dict를 반환한다."""
    cost = calculate_cost(recipe, price_table)
    margin = calculate_margin(cost, price)
    stock_usage = calculate_stock_usage(recipe, price_table)
    status = "PASS" if margin >= target_margin else "FAIL"
    return {
        "cost": cost,
        "price": round(price),
        "margin": margin,
        "stock_usage": stock_usage,
        "target_margin": target_margin,
        "status": status,
    }
