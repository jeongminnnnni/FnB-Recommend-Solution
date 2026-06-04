import json
import re

REQUIRED = ["instruction", "reasoning", "menu", "calculation"]

with open("raw_answers.txt", "r", encoding="utf-8") as f:
    raw = f.read()

# Strip markdown code fences
raw = re.sub(r"```[a-zA-Z]*", "", raw)
raw = raw.replace("```", "")

# Extract top-level balanced [...] arrays
arrays_text = []
depth = 0
start = None
in_str = False
escape = False
for i, ch in enumerate(raw):
    if in_str:
        if escape:
            escape = False
        elif ch == "\\":
            escape = True
        elif ch == '"':
            in_str = False
        continue
    if ch == '"':
        in_str = True
    elif ch == "[":
        if depth == 0:
            start = i
        depth += 1
    elif ch == "]":
        depth -= 1
        if depth == 0 and start is not None:
            arrays_text.append(raw[start:i + 1])
            start = None

valid = []
problems = []
total_extracted = 0

for arr_idx, arr_text in enumerate(arrays_text, 1):
    try:
        arr = json.loads(arr_text)
    except json.JSONDecodeError as e:
        problems.append((f"배열 {arr_idx}", f"JSON 파싱 실패: {e}"))
        continue

    for item_idx, obj in enumerate(arr, 1):
        total_extracted += 1
        label = f"배열{arr_idx}-항목{item_idx}"
        if not isinstance(obj, dict):
            problems.append((label, "dict가 아님"))
            continue
        missing = [k for k in REQUIRED if k not in obj]
        if missing:
            problems.append((label, f"누락 필드: {', '.join(missing)}"))
            continue
        valid.append(obj)

with open("cot_dataset.jsonl", "w", encoding="utf-8") as f:
    for obj in valid:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"추출된 JSON 배열 수: {len(arrays_text)}")
print(f"추출된 총 항목 수: {total_extracted}")
print(f"검증 통과(저장됨): {len(valid)}")
print(f"문제 있음: {len(problems)}")
for label, msg in problems:
    print(f"  - {label}: {msg}")
