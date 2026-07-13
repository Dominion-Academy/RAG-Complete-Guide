def repair_json_string(text: str) -> str:
    new_text = text[:]
    if new_text.startswith("```json"):
        text = text[len("```json") :]
    if new_text.endswith("```"):
        text = text[: -len("```")]
    new_text = text.strip()
    return new_text
