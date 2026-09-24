# TalentMatch: Resume-to-Job Semantic Matching & Ranking Engine

Paste a job description and get ranked candidates, each with a plain-English reason for the match.
Everything runs on **synthetic data** (600 resumes, 40 job descriptions). No scraping and no real personal data.

**Live demo:** https://talentmatch-c7rjvt6jjuuviwznkbwsgk.streamlit.app/

## What it does
1. **Entity extraction (rule-based):** pulls skills (57-skill taxonomy with aliases such as `sklearn`, `k8s`), years of experience and education level from resumes and job descriptions. It ignores lines like "Currently learning X" so they don't count as skills.
2. **Semantic retrieval:** embeds text with `all-MiniLM-L6-v2` and searches a FAISS index for the top 100 candidates per job description.
3. **Re-ranker:** a scikit-learn `GradientBoostingRegressor` blends semantic similarity with required and preferred skill coverage, experience fit and education fit.
4. **Explanations:** every candidate gets a plain-English "why this match", showing matched and missing skills, experience fit and how far the re-ranker moved them versus semantic search alone.

## Evaluation
Measured on 16 held-out synthetic job descriptions with graded relevance labels (0 to 3). Relevant means grade 2 or higher, and NDCG is computed against the ideal ranking over all 600 resumes.

| Metric | Semantic search alone | Semantic + re-ranker |
|---|---|---|
| Precision@5 | 0.562 | 0.988 |
| NDCG@5 | 0.392 | 0.822 |
| Precision@10 | 0.538 | 0.975 |
| NDCG@10 | 0.405 | 0.823 |

- The re-ranker beat semantic-only on NDCG@10 for 16 of 16 held-out job descriptions.
- Extraction quality against hidden ground truth: skill precision 0.996 (excluding deliberately keyword-stuffed resumes) and recall 0.991, with years of experience and education level exact on every resume.
- **Index benchmark (100k jittered vectors):** exact search took 10.4 ms per query and FAISS HNSW took 0.19 ms per query, with recall@10 of 0.998.

## Limitations
- **Synthetic labels.** Relevance labels are derived from the same kinds of signals the re-ranker uses (skill coverage, experience, education), plus noise and a hidden quality factor. The large gain therefore demonstrates the evaluation method, not real-world accuracy.
- **Retrieval ceiling.** Only about 66% of relevant resumes fall inside the FAISS top 100, which caps what any re-ranker can recover.
- **Fixed taxonomy.** Extraction only recognises 57 skills, and job descriptions with none of them fall back mostly to semantic similarity (the app warns you).
- **Fit score.** The "Fit" number is a scaled model score for comparing candidates on one job description, not a percentage match.

## Project structure
```
app.py              Streamlit UI
talentmatch.py      JD parsing, FAISS retrieval, re-ranking, explanations
requirements.txt    Pinned dependencies
artifacts/          Resumes, job descriptions, FAISS index, trained re-ranker, metrics
```
The artifacts were produced by the Colab notebook that generates the synthetic data, trains the re-ranker and runs the evaluation.

## Run locally
Python 3.12 is recommended.
```bash
pip install -r requirements.txt
streamlit run app.py
```
The first run downloads the MiniLM model (about 90 MB).

`scikit-learn` is pinned to 1.6.1 because the saved re-ranker was trained with that version.

## Stack
Python, sentence-transformers (all-MiniLM-L6-v2), FAISS, scikit-learn, pandas, Streamlit
