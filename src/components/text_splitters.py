def split_by_words(text: str, max_words: int = 100) -> list[str]:
    words = text.split()

    if not words:
        return []

    result = []
    for i in range(0, len(words), max_words):
        chunk = words[i : i + max_words]
        result.append(" ".join(chunk))

    return result


def split_by_chars_with_overlap(text: str, max_chars: int, overlap: int = 0) -> list[str]:
    if not text:
        return []
    if max_chars <= 0 or overlap < 0 or overlap >= max_chars:
        raise ValueError("Invalid parameters")

    result = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunk = text[start:end]
        result.append(chunk)
        if end == len(text):
            break
        start += max_chars - overlap

    return result


def split_by_paragraph(text: str, sep: str = "\n\n") -> list[str]:
    if not text:
        return []
    return text.strip().split(sep)


def recursive_split(text: str, max_chars: int, separators: list[str] | None = None, overlap: int = 0) -> list[str]:
    if not text:
        return []

    if separators is None:
        separators = ["\n\n", "\n"]

    if not separators:
        return split_by_chars_with_overlap(text, max_chars, overlap)

    separator, *other_separators = separators
    parts = text.split(separator)

    results = []
    current_chunk = ""
    for part in parts:
        if len(part) > max_chars:
            results.extend(recursive_split(part, max_chars, other_separators, overlap))
        elif len(current_chunk + part) > max_chars:
            results.append(current_chunk)
            current_chunk = part
        else:
            current_chunk = current_chunk + part

    return results
