from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from text_clean import normalize, tokenize_keywords

# load once (cached after first run)
_sbert = SentenceTransformer("all-MiniLM-L6-v2")

def tfidf_score(resume_text: str, jd_text: str) -> float:
    r, j = normalize(resume_text), normalize(jd_text)
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95)
    X = vec.fit_transform([j, r])  # 0 JD, 1 Resume
    return float(cosine_similarity(X[0], X[1])[0][0]) * 100.0

def sbert_score(resume_text: str, jd_text: str) -> float:
    r, j = resume_text.strip(), jd_text.strip()
    embs = _sbert.encode([j, r], normalize_embeddings=True)
    return float((embs[0] * embs[1]).sum()) * 100.0  # cosine via dot (normalized)

def keyword_coverage(resume_text: str, jd_text: str):
    rt = set(tokenize_keywords(normalize(resume_text)))
    jt = set(tokenize_keywords(normalize(jd_text)))
    matched = sorted(jt & rt)
    missing = sorted(jt - rt)
    cov = 0.0 if not jt else round(100.0 * len(matched) / len(jt), 1)
    return cov, matched[:150], missing[:150]

def score_all(resume_text: str, jd_text: str) -> Dict:
    tfidf = round(tfidf_score(resume_text, jd_text), 1)
    sbert = round(sbert_score(resume_text, jd_text), 1)
    coverage, matched, missing = keyword_coverage(resume_text, jd_text)
    return {
        "tfidf_similarity_pct": tfidf,
        "semantic_similarity_pct": sbert,
        "keyword_coverage_pct": coverage,
        "matched_keywords": matched,
        "missing_keywords": missing,
    }
