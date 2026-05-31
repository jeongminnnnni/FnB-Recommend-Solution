"""margin_engine 모듈에 대한 pytest 테스트.

커버 케이스: 정상(PASS) / 경계(==target PASS) / 실패(FAIL) / 재고 활용률 / 예외 / 중간발표 샘플 검산.
"""

import pytest

from margin_engine import (
    UnknownIngredientError,
    calculate_cost,
    calculate_margin,
    calculate_stock_usage,
    evaluate,
)

# 공용 단가표: 1000 단위당 가격으로 표기
PRICE_TABLE = {
    "우유": {"unit_amount": 1000, "unit_price": 1200},   # 1000ml에 1200원
    "원두": {"unit_amount": 1000, "unit_price": 20000},  # 1000g에 20000원
    "시럽": {"unit_amount": 1000, "unit_price": 8000},   # 1000ml에 8000원
}

# 총 재료비 1400원이 되도록 구성한 샘플 레시피
#   우유 500ml -> 600원, 원두 20g -> 400원, 시럽 50ml -> 400원
SAMPLE_RECIPE = [
    {"name": "우유", "amount": 500, "from_stock": False},
    {"name": "원두", "amount": 20, "from_stock": False},
    {"name": "시럽", "amount": 50, "from_stock": False},
]


def test_calculate_cost_sums_ingredient_costs_only():
    """재료비만 합산해 총원가가 1400원으로 나오는지."""
    assert calculate_cost(SAMPLE_RECIPE, PRICE_TABLE) == 1400


def test_normal_case_passes():
    """마진이 목표(70%)를 넘으면 PASS."""
    result = evaluate(SAMPLE_RECIPE, PRICE_TABLE, price=4800)
    assert result["margin"] == 70.8
    assert result["status"] == "PASS"


def test_boundary_case_equal_to_target_passes():
    """마진이 목표와 정확히 같으면(70.0 == 70) PASS."""
    # cost 1500, price 5000 -> margin 70.0
    assert calculate_margin(1500, 5000) == 70.0
    result = evaluate(SAMPLE_RECIPE, PRICE_TABLE, price=5000, target_margin=70)
    # SAMPLE_RECIPE는 cost 1400 -> margin 72.0 이므로, 경계 검증은 calculate_margin으로 별도 확인
    assert result["status"] == "PASS"

    # 정확히 목표와 같은 상황을 evaluate로 직접 구성
    boundary_table = {"단일재료": {"unit_amount": 1, "unit_price": 1500}}
    boundary_recipe = [{"name": "단일재료", "amount": 1, "from_stock": False}]
    boundary = evaluate(boundary_recipe, boundary_table, price=5000, target_margin=70)
    assert boundary["cost"] == 1500
    assert boundary["margin"] == 70.0
    assert boundary["status"] == "PASS"


def test_failure_case_below_target_fails():
    """마진이 목표 미달이면 FAIL."""
    fail_table = {"단일재료": {"unit_amount": 1, "unit_price": 2000}}
    fail_recipe = [{"name": "단일재료", "amount": 1, "from_stock": False}]
    result = evaluate(fail_recipe, fail_table, price=5000, target_margin=70)
    assert result["cost"] == 2000
    assert result["margin"] == 60.0
    assert result["status"] == "FAIL"


def test_stock_usage_with_mixed_ingredients():
    """from_stock 재료가 섞였을 때 재고 활용률이 올바르게 계산되는지."""
    recipe = [
        {"name": "우유", "amount": 500, "from_stock": True},   # 600원 (재고)
        {"name": "원두", "amount": 20, "from_stock": False},   # 400원
        {"name": "시럽", "amount": 50, "from_stock": False},   # 400원
    ]
    # 재고 600 / 전체 1400 * 100 = 42.857... -> 42.9
    assert calculate_stock_usage(recipe, PRICE_TABLE) == 42.9


def test_stock_usage_all_zero_when_no_stock():
    """from_stock 재료가 없으면 활용률은 0.0."""
    assert calculate_stock_usage(SAMPLE_RECIPE, PRICE_TABLE) == 0.0


def test_unknown_ingredient_raises():
    """단가표에 없는 재료를 넣으면 명확한 예외가 발생하는지."""
    recipe = [{"name": "없는재료", "amount": 100, "from_stock": False}]
    with pytest.raises(UnknownIngredientError, match="없는재료"):
        calculate_cost(recipe, PRICE_TABLE)


def test_midterm_sample_check():
    """중간발표 샘플 검산: 원가 1400원, 판매가 4800원 -> 마진 70.8%, PASS."""
    assert calculate_margin(1400, 4800) == 70.8
    result = evaluate(SAMPLE_RECIPE, PRICE_TABLE, price=4800)
    assert result == {
        "cost": 1400,
        "price": 4800,
        "margin": 70.8,
        "stock_usage": 0.0,
        "target_margin": 70,
        "status": "PASS",
    }
