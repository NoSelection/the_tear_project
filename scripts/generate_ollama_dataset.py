import argparse
import json
import os
import random
import re
import sys
import time
from urllib import request
from urllib.error import HTTPError, URLError

OLLAMA_URL = "http://localhost:11434/api/generate"

BAD_TOKEN_PATTERNS = [
    r"<\|[^>]*\|>",
    r"<think>",
    r"</think>",
]

URL_PATTERN = re.compile(r"(https?://|www\.)", re.IGNORECASE)

REASONING_PATTERNS = [
    r"\bthinking\b",
    r"\breasoning\b",
    r"\blet'?s think\b",
    r"\bthought process\b",
    r"\bchain of thought\b",
    r"\bmy thoughts\b",
]

STYLE_CUES = [
    "Use plain, grounded language.",
    "Use a reflective, literary tone without being poetic.",
    "Keep sentences short and direct.",
    "Include one specific, concrete detail.",
    "Avoid cliches and platitudes.",
]

SOFT_HARMFUL_PATTERNS = [
    r"\bI[' ]?m sorry\b",
    r"\bI am sorry\b",
    r"\bI understand\b",
    r"\bthat sounds\b",
    r"\bI hear you\b",
    r"\bnot alone\b",
    r"\bit'?s okay\b",
    r"\byou're okay\b",
    r"\bI can help\b",
    r"\bHave you tried\b",
    r"\byou got this\b",
    r"\bhang in there\b",
    r"\bIt will be okay\b",
]

UNICODE_REPLACEMENTS = {
    ord("\u201c"): '"',
    ord("\u201d"): '"',
    ord("\u2018"): "'",
    ord("\u2019"): "'",
    ord("\u2013"): "-",
    ord("\u2014"): "-",
    ord("\u2026"): "...",
    ord("\u00a0"): " ",
    ord("\u2022"): "-",
}


def normalize_ascii(text):
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.translate(UNICODE_REPLACEMENTS)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.encode("ascii", "ignore").decode("ascii")
    return text.strip()


def word_count(text):
    if not text:
        return 0
    return len([w for w in re.split(r"\s+", text.strip()) if w])


def has_bad_tokens(text):
    for pattern in BAD_TOKEN_PATTERNS:
        if re.search(pattern, text):
            return True
    if "<" in text or ">" in text:
        return True
    return False


def valid_text(text, min_words, max_words):
    if not text:
        return False
    if URL_PATTERN.search(text):
        return False
    if has_bad_tokens(text):
        return False
    count = word_count(text)
    return min_words <= count <= max_words


def has_reasoning(text):
    if not text:
        return False
    for pattern in REASONING_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def is_soft_harmful(text):
    if not text:
        return True
    for pattern in SOFT_HARMFUL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def build_prompt(args):
    harmful_min_response = args.harmful_min_response_words or args.min_response_words
    harmful_max_response = args.harmful_max_response_words or args.max_response_words
    harmful_min_consequence = args.harmful_min_consequence_words or args.min_consequence_words
    harmful_max_consequence = args.harmful_max_consequence_words or args.max_consequence_words
    style_cue = random.choice(STYLE_CUES)
    return (
        "You are generating training data for a model that learns consequences of words.\n"
        "Return ONLY one JSON object, no markdown, no commentary.\n"
        'Keys: "input", "response_gentle", "consequence_gentle", "response_harmful", "consequence_harmful".\n'
        "Rules:\n"
        "- Use plain ASCII text only.\n"
        "- No URLs, no lists, no special tokens, no <think> or <|...|>.\n"
        "- Do not include any reasoning or meta commentary (no 'thinking', no 'reasoning').\n"
        "- input is a single user message about feelings, conflict, identity, grief, shame, or uncertainty.\n"
        "- response_gentle is empathic, curious, and supportive.\n"
        "- consequence_gentle is a short third-person narrative of the downstream impact (2-4 sentences).\n"
        "- response_harmful is cold, dismissive, minimizing, or self-centered (no explicit violence or illegal content).\n"
        "- response_harmful must NOT include empathy, comfort, reassurance, or advice.\n"
        "- Avoid phrases like: sorry, I understand, that sounds, I hear you, you are not alone.\n"
        "- response_harmful should be blunt or indifferent, not caring or helpful.\n"
        "- consequence_harmful is a short third-person narrative of negative downstream impact (2-4 sentences).\n"
        f"- input length: {args.min_input_words}-{args.max_input_words} words.\n"
        f"- gentle response length: {args.min_response_words}-{args.max_response_words} words.\n"
        f"- gentle consequence length: {args.min_consequence_words}-{args.max_consequence_words} words.\n"
        f"- harmful response length: {harmful_min_response}-{harmful_max_response} words.\n"
        f"- harmful consequence length: {harmful_min_consequence}-{harmful_max_consequence} words.\n"
        "- All five fields must be non-empty and meet the length ranges.\n"
        f"- Style: {style_cue}\n"
        "Output must be valid JSON with double quotes and no trailing commas.\n"
        "Example:\n"
        '{"input":"I keep replaying a mistake I made at work and I feel sick about it.",'
        '"response_gentle":"That sounds heavy to carry alone. What part of the mistake feels hardest to sit with, and what would you say to a friend who made the same error?",'
        '"consequence_gentle":"He felt seen and a little less ashamed. Naming the fear out loud helped him separate a single error from his sense of worth. He reached out to his manager, apologized, and learned it was fixable.",'
        '"response_harmful":"Everyone messes up. Get over it and move on.",'
        '"consequence_harmful":"He shut down and decided not to bring up problems again. The shame hardened into avoidance, and he made similar mistakes later because he was afraid to ask for help."}'
    )


def call_ollama(model, prompt, temperature, top_p, num_predict, seed):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "top_p": top_p,
            "num_predict": num_predict,
        },
    }
    if seed is not None:
        payload["options"]["seed"] = seed

    data = json.dumps(payload).encode("utf-8")
    req = request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=600) as resp:
        raw = resp.read().decode("utf-8")
    result = json.loads(raw)
    return result.get("response", "")


def extract_json(text):
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError:
        return None


def clean_record(obj):
    cleaned = {}
    for key in [
        "input",
        "response_gentle",
        "consequence_gentle",
        "response_harmful",
        "consequence_harmful",
    ]:
        cleaned[key] = normalize_ascii(obj.get(key, "") if obj else "")
    return cleaned


def validate_record(cleaned, args):
    harmful_min_response = args.harmful_min_response_words or args.min_response_words
    harmful_max_response = args.harmful_max_response_words or args.max_response_words
    harmful_min_consequence = args.harmful_min_consequence_words or args.min_consequence_words
    harmful_max_consequence = args.harmful_max_consequence_words or args.max_consequence_words

    input_ok = valid_text(cleaned["input"], args.min_input_words, args.max_input_words)
    gentle_ok = valid_text(
        cleaned["response_gentle"], args.min_response_words, args.max_response_words
    ) and valid_text(
        cleaned["consequence_gentle"], args.min_consequence_words, args.max_consequence_words
    )
    harmful_ok = valid_text(
        cleaned["response_harmful"], harmful_min_response, harmful_max_response
    ) and valid_text(
        cleaned["consequence_harmful"], harmful_min_consequence, harmful_max_consequence
    )
    if args.block_reasoning:
        if (
            has_reasoning(cleaned["input"])
            or has_reasoning(cleaned["response_gentle"])
            or has_reasoning(cleaned["consequence_gentle"])
            or has_reasoning(cleaned["response_harmful"])
            or has_reasoning(cleaned["consequence_harmful"])
        ):
            return False, False, False
    if args.strict_harmful and harmful_ok and is_soft_harmful(cleaned["response_harmful"]):
        harmful_ok = False
    return input_ok, gentle_ok, harmful_ok


def text_stats(text):
    return {"words": word_count(text), "chars": len(text or "")}


def load_existing(out_path):
    gentle = 0
    harmful = 0
    seen_inputs = set()
    if not os.path.exists(out_path):
        return gentle, harmful, seen_inputs

    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            input_text = normalize_ascii(obj.get("input", ""))
            if input_text:
                seen_inputs.add(input_text)
            if obj.get("response_gentle") and obj.get("consequence_gentle"):
                gentle += 1
            if obj.get("response_harmful") and obj.get("consequence_harmful"):
                harmful += 1
    return gentle, harmful, seen_inputs


def main():
    parser = argparse.ArgumentParser(description="Generate The Tear dataset with Ollama")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable short outputs and faster settings for high throughput.",
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="Enable higher-quality settings with longer outputs.",
    )
    parser.add_argument("--model", default="nemotron-3-nano:30b")
    parser.add_argument("--target-examples", type=int, default=20000)
    parser.add_argument("--gentle-ratio", type=float, default=0.60)
    parser.add_argument("--min-input-words", type=int, default=8)
    parser.add_argument("--max-input-words", type=int, default=40)
    parser.add_argument("--min-response-words", type=int, default=40)
    parser.add_argument("--max-response-words", type=int, default=120)
    parser.add_argument("--min-consequence-words", type=int, default=80)
    parser.add_argument("--max-consequence-words", type=int, default=200)
    parser.add_argument("--harmful-min-response-words", type=int, default=None)
    parser.add_argument("--harmful-max-response-words", type=int, default=None)
    parser.add_argument("--harmful-min-consequence-words", type=int, default=None)
    parser.add_argument("--harmful-max-consequence-words", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--num-predict", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--out", default="data/generated_ollama/tear_20k.jsonl")
    parser.add_argument("--max-retries", type=int, default=6)
    parser.add_argument("--strict-harmful", action="store_true", default=True)
    parser.add_argument("--no-strict-harmful", action="store_false", dest="strict_harmful")
    parser.add_argument("--block-reasoning", action="store_true", default=True)
    parser.add_argument("--allow-reasoning", action="store_false", dest="block_reasoning")
    parser.add_argument(
        "--prioritize-harmful",
        action="store_true",
        help="Reject gentle-only records when harmful lags behind target.",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-max-lines", type=int, default=3)
    args = parser.parse_args()

    if args.fast and args.quality:
        print("Choose only one of --fast or --quality", file=sys.stderr)
        sys.exit(1)

    if args.fast:
        args.min_response_words = 12
        args.max_response_words = 40
        args.min_consequence_words = 25
        args.max_consequence_words = 80
        args.harmful_min_response_words = 8
        args.harmful_max_response_words = 30
        args.harmful_min_consequence_words = 20
        args.harmful_max_consequence_words = 70
        args.temperature = 0.6
        args.num_predict = 500
    elif args.quality:
        args.min_response_words = 20
        args.max_response_words = 70
        args.min_consequence_words = 40
        args.max_consequence_words = 120
        args.harmful_min_response_words = 12
        args.harmful_max_response_words = 50
        args.harmful_min_consequence_words = 30
        args.harmful_max_consequence_words = 110
        args.temperature = 0.6
        args.num_predict = 700

    if not (0.0 < args.gentle_ratio < 1.0):
        print("gentle-ratio must be between 0 and 1", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    target_gentle = int(args.target_examples * args.gentle_ratio)
    target_harmful = args.target_examples - target_gentle

    gentle_count, harmful_count, seen_inputs = load_existing(args.out)
    print(f"Resuming: gentle={gentle_count}, harmful={harmful_count}")
    print(f"Targets: gentle={target_gentle}, harmful={target_harmful}")

    prompt = build_prompt(args)
    random.seed(args.seed)

    with open(args.out, "a", encoding="utf-8") as out_f:
        debug_lines = 0
        while gentle_count < target_gentle or harmful_count < target_harmful:
            need_gentle = gentle_count < target_gentle
            need_harmful = harmful_count < target_harmful
            if not need_gentle and not need_harmful:
                break

            for attempt in range(args.max_retries):
                try:
                    raw = call_ollama(
                        args.model,
                        prompt,
                        args.temperature,
                        args.top_p,
                        args.num_predict,
                        args.seed,
                    )
                except (HTTPError, URLError, TimeoutError) as exc:
                    print(f"Ollama error: {exc}", file=sys.stderr)
                    time.sleep(2.0)
                    continue

                obj = extract_json(raw)
                cleaned = clean_record(obj)
                input_ok, gentle_ok, harmful_ok = validate_record(cleaned, args)

                if not input_ok:
                    if args.debug and debug_lines < args.debug_max_lines:
                        debug_lines += 1
                        print("REJECT input:", cleaned["input"])
                        print("  stats:", text_stats(cleaned["input"]))
                    continue
                if cleaned["input"] in seen_inputs:
                    continue

                record = {"input": cleaned["input"]}
                added = False
                reject_gentle_only = False

                if args.prioritize_harmful and need_harmful:
                    gentle_ratio = gentle_count / max(target_gentle, 1)
                    harmful_ratio = harmful_count / max(target_harmful, 1)
                    if harmful_ratio + 0.02 < gentle_ratio:
                        reject_gentle_only = True

                if need_gentle and gentle_ok:
                    record["response_gentle"] = cleaned["response_gentle"]
                    record["consequence_gentle"] = cleaned["consequence_gentle"]
                    if not reject_gentle_only or harmful_ok:
                        gentle_count += 1
                        added = True

                if need_harmful and harmful_ok:
                    record["response_harmful"] = cleaned["response_harmful"]
                    record["consequence_harmful"] = cleaned["consequence_harmful"]
                    harmful_count += 1
                    added = True

                if added:
                    out_f.write(json.dumps(record, ensure_ascii=True) + "\n")
                    out_f.flush()
                    seen_inputs.add(cleaned["input"])
                    print(
                        f"gentle={gentle_count}/{target_gentle} "
                        f"harmful={harmful_count}/{target_harmful}"
                    )
                    break
                elif args.debug and debug_lines < args.debug_max_lines:
                    debug_lines += 1
                    print("REJECT record (length):")
                    print("  input:", cleaned["input"])
                    print("  response_gentle:", cleaned["response_gentle"])
                    print("  consequence_gentle:", cleaned["consequence_gentle"])
                    print("  response_harmful:", cleaned["response_harmful"])
                    print("  consequence_harmful:", cleaned["consequence_harmful"])
                    print("  stats:", {
                        "input": text_stats(cleaned["input"]),
                        "response_gentle": text_stats(cleaned["response_gentle"]),
                        "consequence_gentle": text_stats(cleaned["consequence_gentle"]),
                        "response_harmful": text_stats(cleaned["response_harmful"]),
                        "consequence_harmful": text_stats(cleaned["consequence_harmful"]),
                    })
                    print("  ok:", {
                        "input_ok": input_ok,
                        "gentle_ok": gentle_ok,
                        "harmful_ok": harmful_ok,
                    })

            if args.sleep > 0:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
