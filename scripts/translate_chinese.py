"""
Translate Chinese text to English using DeepSeek API.
Preprocessing step: run before training to translate Chinese datasets (e.g., Weibo-21)
so that English-only models (RoBERTa, emotion classifier) can process them effectively.
"""
import json, os, time, argparse
from openai import OpenAI


def translate_text(client, text: str, model: str = "deepseek-chat") -> str:
    """Translate a single Chinese text to English. Returns original if empty."""
    if not text or not text.strip():
        return text
    # Quick check: if mostly ASCII, skip
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    if ascii_chars / max(len(text), 1) > 0.7:
        return text
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "system",
                "content": "Translate the following Chinese text to English. Output ONLY the translation, no explanations."
            }, {
                "role": "user",
                "content": text
            }],
            temperature=0.1,
            max_tokens=min(len(text) * 3 + 50, 2000)
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  Translate error: {e}, keeping original")
        return text


def translate_dataset(json_path: str, output_path: str,
                      text_key: str = "text", model: str = "deepseek-chat"):
    """Translate all news texts in a dataset JSON file."""
    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com"
    )

    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    translated = 0
    for i, item in enumerate(data):
        raw = item.get(text_key, "")
        if raw:
            translated_text = translate_text(client, raw, model=model)
            if translated_text != raw:
                item[text_key] = translated_text
                translated += 1
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(data)} ({translated} translated) ...")
            time.sleep(0.5)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Done: {translated}/{len(data)} items translated → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate Chinese dataset to English")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file")
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--model", type=str, default="deepseek-chat")
    args = parser.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        print("ERROR: Set DEEPSEEK_API_KEY environment variable first.")
        exit(1)

    translate_dataset(args.input, args.output, args.text_key, args.model)
