def split_by_words(text: str, max_words: int = 100) -> list[str]:
    words = text.split()

    if not words:
        return []

    result = []
    for i in range(0, len(words), max_words):
        chunk = words[i : i + max_words]
        result.append(" ".join(chunk))

    return result
