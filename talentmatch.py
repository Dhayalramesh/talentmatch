"""TalentMatch core: JD parsing, FAISS retrieval, re-ranking and plain-English explanations.

Everything here mirrors the Colab notebook so the saved re-ranker sees the same features it was trained on.
"""
import json
import re
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd

ART = Path(__file__).parent / "artifacts"

# ---------------------------------------------------------------- skill taxonomy (same as notebook)
SKILL_ALIASES = {
    "python": ["python"], "sql": ["sql"], "pandas": ["pandas"], "numpy": ["numpy"],
    "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
    "machine learning": ["machine learning", "ml models"],
    "deep learning": ["deep learning", "neural networks"],
    "pytorch": ["pytorch"], "tensorflow": ["tensorflow"],
    "nlp": ["nlp", "natural language processing"],
    "transformers": ["transformers", "hugging face", "huggingface"],
    "llm": ["llm", "llms", "large language models"],
    "faiss": ["faiss"], "spacy": ["spacy"],
    "computer vision": ["computer vision", "opencv"],
    "feature engineering": ["feature engineering"],
    "statistics": ["statistics", "statistical modelling", "statistical modeling"],
    "a/b testing": ["a/b testing", "ab testing", "a/b tests"],
    "mlflow": ["mlflow"], "streamlit": ["streamlit"],
    "spark": ["spark", "pyspark"], "airflow": ["airflow"], "kafka": ["kafka"],
    "dbt": ["dbt"], "snowflake": ["snowflake"], "etl": ["etl"],
    "postgresql": ["postgresql", "postgres"], "mysql": ["mysql"],
    "mongodb": ["mongodb"], "redis": ["redis"],
    "aws": ["aws", "amazon web services"], "gcp": ["gcp", "google cloud"], "azure": ["azure"],
    "docker": ["docker"], "kubernetes": ["kubernetes", "k8s"], "terraform": ["terraform"],
    "jenkins": ["jenkins"], "ci/cd": ["ci/cd", "cicd", "ci-cd"], "linux": ["linux"], "git": ["git"],
    "prometheus": ["prometheus"], "grafana": ["grafana"],
    "tableau": ["tableau"], "power bi": ["power bi", "powerbi"], "excel": ["excel"],
    "java": ["java"], "spring boot": ["spring boot", "springboot"],
    "django": ["django"], "flask": ["flask"], "fastapi": ["fastapi"],
    "rest apis": ["rest api", "rest apis", "restful apis"], "microservices": ["microservices"],
    "javascript": ["javascript"], "typescript": ["typescript"],
    "react": ["react", "react.js", "reactjs"], "node.js": ["node.js", "nodejs"], "css": ["css", "css3"],
}
ALIAS2CANON = {a.lower(): c for c, al in SKILL_ALIASES.items() for a in al}

_aliases = sorted(ALIAS2CANON, key=len, reverse=True)
SKILL_RE = re.compile(r"(?<![A-Za-z0-9+#])(" + "|".join(re.escape(a) for a in _aliases) + r")(?![A-Za-z0-9+#])", re.I)
LEARNING_RE = re.compile(r"currently\s+(?:learning|studying|exploring)", re.I)
EDU_PATTERNS = [
    (3, re.compile(r"\bph\.?\s?d\b|\bdoctorate\b", re.I)),
    (2, re.compile(r"\bm\.?\s?tech\b|\bm\.?\s?sc\b|\bmca\b|\bmba\b|\bmaster(?:'?s)?\b", re.I)),
    (1, re.compile(r"\bb\.?\s?tech\b|\bb\.e\b|\bb\.?\s?sc\b|\bbca\b|\bb\.?\s?com\b|\bbachelor(?:'?s)?\b", re.I)),
]
REQ_HEAD = re.compile(r"^(required|requirements?|qualifications|must have|what you.?ll need|what we.?re looking for)\b", re.I)
PREF_HEAD = re.compile(r"^(nice to have|preferred|good to have|bonus|plus points?)\b", re.I)


def extract_skills(text):
    found = set()
    for line in text.splitlines():
        if LEARNING_RE.search(line):
            continue
        for m in SKILL_RE.finditer(line):
            found.add(ALIAS2CANON[m.group(1).lower()])
    return sorted(found)


PLUS_RE = re.compile(r"\b(?:is a plus|are a plus|a plus|nice to have|preferred|bonus|good to have)\b", re.I)


def _sentences(lines):
    """Split lines into sentences, dropping any that describe a degree
    ('...Computer Science, Statistics or a related field' names fields of study, not skills)."""
    kept = []
    for line in lines:
        for sent in re.split(r"(?<=[.!?])\s+", line):
            if not re.search(r"\bdegree\b", sent, re.I):
                kept.append(sent)
    return kept


def extract_jd(text):
    """Parse a job description. Uses 'Required' / 'Nice to have' sections when present;
    for free-form text, every recognised skill counts as required unless the sentence says it is a plus."""
    section, req_lines, pref_lines, saw_heading = None, [], [], False
    for line in text.splitlines():
        low = line.strip().lower().lstrip("#*- ").strip()
        if REQ_HEAD.match(low) and len(low) < 60:
            section, saw_heading = "req", True
            rest = line.split(":", 1)[1] if ":" in line else ""
            if rest.strip():
                req_lines.append(rest)
            continue
        if PREF_HEAD.match(low) and len(low) < 60:
            section, saw_heading = "pref", True
            rest = line.split(":", 1)[1] if ":" in line else ""
            if rest.strip():
                pref_lines.append(rest)
            continue
        if section == "req":
            req_lines.append(line)
        elif section == "pref":
            pref_lines.append(line)

    if saw_heading:
        req_skill_lines, pref_skill_lines = _sentences(req_lines), pref_lines
        exp_txt = "\n".join(req_lines)
    else:  # free-form paragraph JD
        sents = _sentences(text.splitlines())
        pref_skill_lines = [x for x in sents if PLUS_RE.search(x)]
        req_skill_lines = [x for x in sents if not PLUS_RE.search(x)]
        exp_txt = text

    lo, hi = 0, 99
    m = re.search(r"(\d{1,2})\s*(?:-|–|to)\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?)", exp_txt, re.I)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)", exp_txt, re.I)
        if m:
            lo = int(m.group(1))
    levels = [lvl for lvl, pat in EDU_PATTERNS if pat.search(exp_txt)]
    return dict(required=extract_skills("\n".join(req_skill_lines)),
                preferred=extract_skills("\n".join(pref_skill_lines)),
                min_years=lo, max_years=hi, min_edu=min(levels) if levels else 0)


# ---------------------------------------------------------------- features + explanations (same as notebook)
def pair_features(jd, res, sem, sem_z):
    req, pref, sk = set(jd["required"]), set(jd["preferred"]), set(res["skills"])
    gap = max(jd["min_years"] - res["years"], res["years"] - jd["max_years"], 0)
    return dict(sem_sim=sem, sem_z=sem_z,
                req_cov=len(req & sk) / max(1, len(req)), pref_cov=len(pref & sk) / max(1, len(pref)),
                n_missing_req=len(req - sk), exp_fit=max(0.0, 1 - gap / 4), years=res["years"],
                edu_gap=res["edu_level"] - jd["min_edu"], n_skills=len(sk))


def explain(jd, res, sem_rank, top_k):
    req, pref, sk = jd["required"], jd["preferred"], set(res["skills"])
    hit = [s for s in req if s in sk]
    miss = [s for s in req if s not in sk]
    parts = []
    if req:
        s = f"Matches {len(hit)} of {len(req)} required skills" + (f" ({', '.join(hit)})" if hit else "")
        if miss:
            s += f"; missing {', '.join(miss)}"
        parts.append(s + ".")
    ph = [s for s in pref if s in sk]
    if pref and ph:
        parts.append(f"Also has nice-to-haves: {', '.join(ph)}.")
    y, lo, hi = res["years"], jd["min_years"], jd["max_years"]
    yr = f"{y:g} yr" + ("" if y == 1 else "s")
    if y < lo:
        parts.append(f"{yr} experience is {lo - y:g} below the {lo}-year minimum.")
    elif y > hi and hi < 99:
        parts.append(f"{yr} experience is above the {lo}-{hi} year range (may be overqualified).")
    else:
        parts.append(f"{yr} experience fits the requested range.")
    if jd["min_edu"]:
        names = {1: "Bachelor's", 2: "Master's", 3: "PhD"}
        parts.append(("Education meets the requirement" if res["edu_level"] >= jd["min_edu"]
                      else f"Education is below the requested {names[jd['min_edu']]}") + ".")
    parts.append(f"Semantic match rank: #{int(sem_rank)} of {top_k} by text similarity.")
    return " ".join(parts)


# ---------------------------------------------------------------- engine
class TalentMatch:
    def __init__(self):
        self.cfg = json.loads((ART / "config_and_metrics.json").read_text())
        self.features = self.cfg["features"]
        self.top_k = int(self.cfg["top_k"])
        self.resumes = pd.read_csv(ART / "resumes.csv")
        self.jds = pd.read_csv(ART / "job_descriptions.csv")
        # extracted entities were saved by the notebook (skills / years / education level)
        self.ents = [dict(skills=json.loads(s), years=float(y), edu_level=int(e))
                     for s, y, e in zip(self.resumes.skills, self.resumes.years_extracted,
                                        self.resumes.edu_level_extracted)]
        self.index = faiss.read_index(str(ART / "resumes.faiss"))
        self.reranker = joblib.load(ART / "reranker.joblib")

        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.model.max_seq_length = 256

    def embed(self, texts):
        return np.ascontiguousarray(
            self.model.encode(list(texts), batch_size=64, normalize_embeddings=True,
                              show_progress_bar=False).astype("float32"))

    def rank(self, jd_text, top_n=10, use_reranker=True):
        jd = extract_jd(jd_text)
        sims, ids = self.index.search(self.embed([jd_text]), self.top_k)
        sims, ids = sims[0], ids[0]
        z = (sims - sims.mean()) / (sims.std() + 1e-9)
        rows = []
        for r, (i, s, zz) in enumerate(zip(ids, sims, z)):
            res = self.ents[i]
            rows.append(dict(res_idx=int(i), sem_rank=r + 1, **pair_features(jd, res, float(s), float(zz))))
        df = pd.DataFrame(rows)
        raw = self.reranker.predict(df[self.features]) if use_reranker else df["sem_sim"].values
        df["score"] = raw
        df["fit"] = (np.clip(raw / 3.0, 0, 1) * 100).round(0) if use_reranker else (np.clip(raw, 0, 1) * 100).round(0)
        df = df.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
        df["rank"] = df.index + 1
        out = []
        for r in df.itertuples():
            res, meta = self.ents[r.res_idx], self.resumes.iloc[r.res_idx]
            sk = set(res["skills"])
            out.append(dict(
                rank=r.rank, resume_id=meta.resume_id, name=meta["name"], title=meta.title,
                fit=int(r.fit), sem_sim=r.sem_sim, sem_rank=r.sem_rank, years=res["years"],
                matched=[s for s in jd["required"] if s in sk],
                missing=[s for s in jd["required"] if s not in sk],
                bonus=[s for s in jd["preferred"] if s in sk],
                why=explain(jd, res, r.sem_rank, self.top_k), text=meta.text))
        return jd, out
