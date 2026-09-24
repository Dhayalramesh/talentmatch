import pandas as pd
import streamlit as st

from talentmatch import TalentMatch

st.set_page_config(page_title="TalentMatch", page_icon="🎯", layout="wide")

DEFAULT_JD = """Senior NLP Engineer (Bengaluru)

About the role
Build language understanding systems for search and information extraction.

Required
- 4-8 years of experience
- Bachelor's degree in Computer Science or a related field
- Hands-on experience with Python, PyTorch, Transformers, FAISS, spaCy

Nice to have
- Exposure to FastAPI, Docker
"""
EDU = {0: "not specified", 1: "Bachelor's", 2: "Master's", 3: "PhD"}


@st.cache_resource(show_spinner="Loading embedding model and FAISS index (first load takes a minute)...")
def load_engine():
    return TalentMatch()


tm = load_engine()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("About")
    st.write(
        "TalentMatch ranks candidates for a job description in three steps: "
        "**1)** rule-based entity extraction (skills, years, education), "
        "**2)** semantic search with MiniLM embeddings + FAISS, "
        "**3)** a scikit-learn re-ranker that blends semantic similarity with structured signals."
    )
    st.caption("All 600 resumes are synthetic. No real personal data is used.")
    st.subheader("Offline evaluation")
    metrics = pd.DataFrame(tm.cfg["metrics"]).drop(columns=["Change"]).round(3)
    metrics.columns = ["Semantic only", "+ Re-ranker"]
    st.dataframe(metrics, width="stretch")
    st.caption(
        "16 held-out synthetic job descriptions, graded relevance labels. "
        "Labels are synthetic, so treat these as a demonstration of the evaluation method, not real-world accuracy."
    )

# ------------------------------------------------------------------ input
st.title("TalentMatch")
st.write("Paste a job description and get ranked candidates, each with a plain-English reason for the match.")

if "jd_text" not in st.session_state:
    st.session_state.jd_text = DEFAULT_JD

NONE = "(paste your own below)"
labels = [f"{r.jd_id} | {r.title}" for r in tm.jds.itertuples()]


def load_sample():
    label = st.session_state.sample
    if label != NONE:
        jd_id = label.split(" | ")[0]
        st.session_state.jd_text = tm.jds.loc[tm.jds.jd_id == jd_id, "text"].iloc[0]


st.selectbox("Load a sample job description", [NONE] + labels, key="sample", on_change=load_sample)
st.text_area("Job description", key="jd_text", height=260)

c1, c2, c3 = st.columns([2, 2, 1])
mode = c1.radio("Ranking method", ["Semantic + re-ranker", "Semantic search only"], horizontal=True)
top_n = c2.slider("Candidates to show", 3, 20, 8)
if c3.button("Rank candidates", type="primary", width="stretch"):
    st.session_state.ran = True

# ------------------------------------------------------------------ results
if st.session_state.get("ran"):
    jd_text = st.session_state.jd_text
    if not jd_text.strip():
        st.warning("Paste a job description first.")
        st.stop()

    use_rr = mode == "Semantic + re-ranker"
    jd, results = tm.rank(jd_text, top_n=top_n, use_reranker=use_rr)

    st.subheader("How the job description was parsed")
    p1, p2, p3 = st.columns(3)
    p1.markdown("**Required skills**  \n" + (" ".join(f":blue-badge[{s}]" for s in jd["required"]) or "none recognised"))
    p2.markdown("**Nice to have**  \n" + (" ".join(f":gray-badge[{s}]" for s in jd["preferred"]) or "none recognised"))
    yrs = "not specified" if jd["min_years"] == 0 and jd["max_years"] == 99 else (
        f"{jd['min_years']}+ years" if jd["max_years"] == 99 else f"{jd['min_years']}-{jd['max_years']} years")
    p3.markdown(f"**Experience**  \n{yrs}  \n**Education**  \n{EDU[jd['min_edu']]}")
    if not jd["required"]:
        st.warning(
            "No skills from the 57-skill taxonomy were found in this job description, so ranking leans almost "
            "entirely on semantic similarity. Try naming skills such as Python, SQL, Docker or React."
        )

    st.subheader(f"Top {len(results)} candidates")
    for r in results:
        with st.container(border=True):
            h1, h2 = st.columns([5, 1])
            yrs_txt = f"{r['years']:g} yr" + ("" if r["years"] == 1 else "s")
            h1.markdown(f"**#{r['rank']} {r['name']}**  \n{r['title']} · {yrs_txt} · {r['resume_id']}")
            h2.metric("Fit", f"{r['fit']}")
            badges = [f":green-badge[{s}]" for s in r["matched"]] + [f":red-badge[missing: {s}]" for s in r["missing"]] \
                + [f":gray-badge[+ {s}]" for s in r["bonus"]]
            if badges:
                st.markdown(" ".join(badges))
            st.write(r["why"])
            if use_rr:
                st.caption(f"Semantic search alone would have placed this candidate #{r['sem_rank']}; the re-ranker put them #{r['rank']}.")
            with st.expander("View synthetic resume"):
                st.text(r["text"])
