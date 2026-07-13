from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi


texts = [
    "We propose a new deep learning architecture for time series forecasting that combines LSTM and attention mechanisms. Our model outperforms existing methods on three public datasets by 12% in terms of RMSE.",
    "This paper investigates the use of graph neural networks for protein-protein interaction prediction. We introduce a novel edge-conditioned convolution and achieve state-of-the-art accuracy on the STRING database.",
    "In this work, we explore reinforcement learning for robotic manipulation in cluttered environments. Our agent learns a policy from raw visual input and generalizes to unseen objects with 85% success rate.",
    "We present a federated learning framework for healthcare data that preserves privacy while maintaining model performance. Differential privacy is applied with a noise budget of epsilon=0.1, achieving 94% of the centralized baseline.",
    "A survey of transformer-based models for natural language understanding is provided. We compare BERT, RoBERTa, and ELECTRA on GLUE and SuperGLUE, showing that ELECTRA is the most efficient.",
]

stemmer = PorterStemmer()


def get_words(text: str) -> list[str]:
    words = text.lower().split()
    words = [stemmer.stem(tok) for tok in words if tok.isalpha()]
    return words


text_words = [get_words(text) for text in texts]
bm25 = BM25Okapi(text_words)

query = "BERT transformer understanding"
query_words = get_words(query)

scores = bm25.get_scores(query_words)
print(scores)

top_docs = bm25.get_top_n(query_words, texts, n=2)
print(top_docs)
