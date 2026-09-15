import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def load_dataset():
    with open("data/jobs.json", "r") as f:
        jobs = json.load(f)
    with open("data/resumes.json", "r") as f:
        resumes = json.load(f)
    return jobs, resumes


def format_for_embedding(record, is_job=True):
    if is_job:
        return f"Role: {record['title']}. Required Skills: {record['skills']}. Responsibilities: {record['summary']}"
    else:
        return f"Current Role: {record['current_title']}. Technical Competencies: {record['skills']}. Background: {record['summary']}"


def main():
    print("1. Loading datasets...")
    jobs, resumes = load_dataset()
    print(f"   Loaded {len(jobs)} JDs and {len(resumes)} Resumes.")

    print("\n2. Initializing Sentence-Transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("\n3. Converting text to semantic embeddings...")
    job_texts = [format_for_embedding(j, is_job=True) for j in jobs]
    resume_texts = [format_for_embedding(r, is_job=False) for r in resumes]

    job_embeddings = model.encode(job_texts, normalize_embeddings=True)
    resume_embeddings = model.encode(resume_texts, normalize_embeddings=True)

    print("\n4. Calculating similarity scores...")
    similarity_matrix = cosine_similarity(resume_embeddings, job_embeddings)

    print("\n================ TOP 3 MATCHES PER CANDIDATE ================")
    TOP_K = 3

    for cand_idx, candidate in enumerate(resumes):
        scores = similarity_matrix[cand_idx]
        ranked_indices = np.argsort(scores)[::-1]

        print(f"\nCandidate: {candidate['name']} ({candidate['current_title']})")
        print("-" * 55)

        for rank in range(TOP_K):
            job_idx = ranked_indices[rank]
            score_pct = scores[job_idx] * 100
            matched_job = jobs[job_idx]
            print(f"  Rank {rank + 1}: [{score_pct:5.1f}% Match] {matched_job['title']} ({matched_job['id']})")


if __name__ == "__main__":
    main()