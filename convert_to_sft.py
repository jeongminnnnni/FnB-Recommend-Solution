import json
from pathlib import Path

SRC = Path("cot_dataset.jsonl")
DST = Path("sft_dataset.jsonl")

# ── 읽기 ──────────────────────────────────────────────────────────────────────
src_lines = [l for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
src_records = [json.loads(l) for l in src_lines]

# ── 변환 ──────────────────────────────────────────────────────────────────────
sft_records = []
for rec in src_records:
    output_obj = {
        "reasoning":   rec["reasoning"],
        "menu":        rec["menu"],
        "calculation": rec["calculation"],
    }
    sft_records.append({
        "instruction": rec["instruction"],
        "output":      json.dumps(output_obj, ensure_ascii=False),
    })

# ── 저장 ──────────────────────────────────────────────────────────────────────
with DST.open("w", encoding="utf-8") as f:
    for rec in sft_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ── 검증 1: 개수 일치 ─────────────────────────────────────────────────────────
print(f"변환 전: {len(src_records)}개  →  변환 후: {len(sft_records)}개  "
      f"{'[일치]' if len(src_records) == len(sft_records) else '[불일치!]'}")

# ── 검증 2: output 문자열이 json.loads로 재파싱 가능한지 ─────────────────────
sample = sft_records[0]
reparsed = json.loads(sample["output"])
keys_ok = set(reparsed.keys()) == {"reasoning", "menu", "calculation"}
print(f"\n[샘플 재파싱 검증]")
print(f"  json.loads 성공: True")
print(f"  최상위 키: {list(reparsed.keys())}  {'[OK]' if keys_ok else '[키 불일치!]'}")
print(f"  calculation 키: {list(reparsed['calculation'].keys())}")

# ── 검증 3: 첫 줄 사람이 읽기 좋게 출력 ──────────────────────────────────────
print(f"\n[첫 번째 줄 (사람이 읽기 좋게)]")
first = sft_records[0]
print(f"  instruction : {first['instruction']}")
output_pretty = json.dumps(json.loads(first["output"]), ensure_ascii=False, indent=4)
print(f"  output (파싱 후):\n{output_pretty}")
