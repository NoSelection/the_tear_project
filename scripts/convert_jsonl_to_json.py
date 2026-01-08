import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Convert JSONL to JSON array")
    parser.add_argument("--in", dest="in_path", required=True)
    parser.add_argument("--out", dest="out_path", required=True)
    args = parser.parse_args()

    if not os.path.exists(args.in_path):
        print(f"Input not found: {args.in_path}", file=sys.stderr)
        sys.exit(1)

    records = []
    with open(args.in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=True, indent=2)

    print(f"Wrote {len(records)} records to {args.out_path}")


if __name__ == "__main__":
    main()
