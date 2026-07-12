import numpy as np


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1.ndim != 1 or vec2.ndim != 1:
        raise ValueError("Векторы должны быть одномерными")
    vec1 = vec1.reshape(1, -1)
    vec2 = vec2.reshape(1, -1)
    vec1_norm = np.linalg.norm(vec1, axis=1, keepdims=True)
    vec2_norm = np.linalg.norm(vec2, axis=1, keepdims=True)
    return float((np.dot(vec1, vec2.T) / np.dot(vec1_norm, vec2_norm))[0, 0])
