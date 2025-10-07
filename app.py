# app.py — minimal public demo
import re
import streamlit as st
import pdfplumber, docx2txt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Job–Candidate Matcher", layout="wide")
st.title("🔍 Job–Candidate Matching (Resume ↔ JD)")

def read_text(uploaded):
    if not uploaded:
        return ""
    name = uploaded.name.lower()
    if name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(uploaded) as pdf:
            for p in pdf.pages:
                text += p.extract_text() or ""
        return text
    if name.endswith(".docx"):
        # docx2txt needs a path; write to tmp
        data = uploaded.read()
        tmp = "/tmp/_in.docx"
        with open(tmp, "wb") as f: f.write(data)
        return docx2txt.process(tmp) or ""
    # .txt or others
    try:
        return uploaded.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""

def clean(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

col1, col2 = st.columns(2)
with col1:
    st.subheader("Job Description")
    jd_file = st.file_uploader("Upload JD (.pdf/.docx/.txt)", type=["pdf","docx","txt"], key="jd")
    jd_text = st.text_area("…or paste JD", height=220)
    if jd_file and not jd_text.strip():
        jd_text = read_text(jd_file)
    jd_text = clean(jd_text)

with col2:
    st.subheader("Resume")
    rs_file = st.file_uploader("Upload Resume (.pdf/.docx/.txt)", type=["pdf","docx","txt"], key="rs")
    rs_text = st.text_area("…or paste Resume", height=220)
    if rs_file and not rs_text.strip():
        rs_text = read_text(rs_file)
    rs_text = clean(rs_text)

st.divider()

if (jd_text and rs_text) and st.button("⚖️ Evaluate Match"):
    vec = TfidfVectorizer(stop_words="english", max_features=6000, ngram_range=(1,2))
    tfidf = vec.fit_transform([jd_text, rs_text])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    pct = round(score*100, 2)

    st.subheader("Overall Match")
    st.progress(min(int(pct), 100))
    st.metric("Similarity Score", f"{pct}%")

    # quick keyword coverage from JD
    vec2 = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1,2))
    X = vec2.fit_transform([jd_text])
    vocab = vec2.get_feature_names_out()
    weights = X.toarray()[0]
    kw = [w for w,_ in sorted(zip(vocab, weights), key=lambda x: x[1], reverse=True)[:30]]

    covered = [k for k in kw if k.lower() in rs_text.lower()]
    missing = [k for k in kw if k.lower() not in rs_text.lower()]

    st.subheader("Top JD Keywords")
    st.write(", ".join(kw))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**✅ Covered in Resume**")
        st.write(", ".join(covered) if covered else "_No strong matches found._")
    with c2:
        st.markdown("**❌ Missing / Low-Signal**")
        st.write(", ".join(missing) if missing else "_Great coverage!_")

    st.subheader("Recommendations")
    tips = []
    if pct < 75: tips.append("Tailor your summary to mirror high-signal JD skills/phrases.")
    if missing: tips.append("Add concrete bullets featuring: " + ", ".join(missing[:8]) + ".")
    if "project" not in rs_text.lower(): tips.append("Include 2–3 quantified ML projects (data size, accuracy, deployment).")
    if "git" not in rs_text.lower(): tips.append("Link a GitHub repo with relevant work.")
    if not tips: tips.append("Looks strong — consider adding metrics (latency, ROI, accuracy deltas).")
    for t in tips: st.markdown(f"- {t}")

st.caption("Keep this file at repo root as `app.py` and list deps in `requirements.txt`.")
