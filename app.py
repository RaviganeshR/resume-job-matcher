import io
import re
import json
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader

app = FastAPI(title="TalentVector AI Suite")

# ==============================================================================
# DOCUMENT PARSER: PDF & WORD DOCS ONLY (.pdf, .docx, .doc)
# ==============================================================================
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}

def extract_text_from_file_bytes(contents: bytes, filename: str) -> str:
    """
    Extracts plain text strictly from PDF and Microsoft Word (.docx, .doc) files.
    """
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Invalid format '{ext}'. Only PDF (.pdf) and Word documents (.docx, .doc) are supported.")

    extracted_text = ""

    # 1. PDF Parser via pypdf
    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(contents))
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    extracted_text += t + " "
        except Exception as e:
            raise ValueError(f"Could not read PDF file: {str(e)}")

    # 2. DOCX / DOC Parser via XML / ZipStream
    elif ext in {".docx", ".doc"}:
        try:
            with zipfile.ZipFile(io.BytesIO(contents)) as docx_zip:
                xml_content = docx_zip.read("word/document.xml")
                tree = ET.fromstring(xml_content)
                namespaces = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
                paragraphs = []
                for p in tree.iter(f"{{{namespaces['w']}}}p"):
                    texts = [node.text for node in p.iter(f"{{{namespaces['w']}}}t") if node.text]
                    if texts:
                        paragraphs.append("".join(texts))
                extracted_text = " ".join(paragraphs)
        except Exception:
            # Fallback for plain formatted legacy doc streams
            try:
                extracted_text = contents.decode("utf-8", errors="ignore")
            except Exception as e:
                raise ValueError(f"Could not read Word document: {str(e)}")

    cleaned_text = " ".join(extracted_text.split())
    if not cleaned_text:
        raise ValueError(f"The document '{filename}' contains no readable text.")

    return cleaned_text

# ==============================================================================
# USE CASE 1: Semantic Matching Pipeline (MiniLM + Cosine Math)
# ==============================================================================
with open("data/jobs.json", "r") as f:
    jobs = json.load(f)

with open("data/resumes.json", "r") as f:
    resumes = json.load(f)

print("Loading SentenceTransformer model ('all-MiniLM-L6-v2')...")
model = SentenceTransformer("all-MiniLM-L6-v2")


def format_job_text(job):
    return f"Role: {job['title']}. Required Skills: {job['skills']}. Responsibilities: {job['summary']}"


def format_resume_text(resume):
    return f"Candidate: {resume['name']}. Current Role: {resume['current_title']}. Technical Competencies: {resume['skills']}. Background: {resume['summary']}"


job_texts = [format_job_text(j) for j in jobs]
job_embeddings = model.encode(job_texts, normalize_embeddings=True)


class UpdateCandidateRequest(BaseModel):
    name: str
    current_title: str
    skills: str
    summary: str


class AddJobRequest(BaseModel):
    id: str
    title: str
    skills: str
    summary: str


@app.get("/api/data")
def get_data():
    return {"jobs": jobs, "resumes": resumes}


@app.get("/api/match/{job_id}")
def match_job(job_id: str):
    global job_embeddings
    if not resumes:
        return {"job": None, "rankings": []}

    job_idx = next((idx for idx, j in enumerate(jobs) if j["id"] == job_id), None)
    if job_idx is None:
        raise HTTPException(status_code=404, detail="Job not found")

    selected_job_vector = job_embeddings[job_idx].reshape(1, -1)
    cand_texts = [format_resume_text(r) for r in resumes]
    cand_embeddings = model.encode(cand_texts, normalize_embeddings=True)
    raw_scores = cosine_similarity(cand_embeddings, selected_job_vector).flatten()

    results = []
    for idx, cand in enumerate(resumes):
        raw_score = float(raw_scores[idx])
        calibrated = max(0.0, min(1.0, (raw_score - 0.20) / 0.65))
        pct = round(calibrated * 100, 1)

        if pct >= 80:
            verdict = "Strong Match"
        elif pct >= 60:
            verdict = "Transferable"
        elif pct >= 45:
            verdict = "Moderate"
        else:
            verdict = "Low Alignment"

        results.append({
            "candidate": cand,
            "raw_cosine": round(raw_score, 3),
            "match_percentage": pct,
            "verdict": verdict
        })

    results.sort(key=lambda x: x["match_percentage"], reverse=True)
    return {"job": jobs[job_idx], "rankings": results}


@app.post("/api/upload-pdf-resume")
async def upload_resume_any_format(
    file: UploadFile = File(...),
    name: str = Form(None),
    title: str = Form(None)
):
    """Universal Upload for Tab 1 (Matches all formats into vector space)."""
    contents = await file.read()
    try:
        extracted_text = extract_text_from_file_bytes(contents, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    cand_id = f"CAND_{len(resumes) + 1:02d}"
    clean_filename = re.sub(r"\.[^.]+$", "", file.filename)
    candidate_name = name.strip() if name and name.strip() else clean_filename
    candidate_title = title.strip() if title and title.strip() else "Applicant Profile"

    new_candidate = {
        "id": cand_id,
        "name": candidate_name,
        "current_title": candidate_title,
        "skills": f"Extracted from {file.filename}",
        "summary": extracted_text[:600] + ("..." if len(extracted_text) > 600 else "")
    }

    resumes.append(new_candidate)
    return {"status": "success", "candidate": new_candidate}


@app.put("/api/resumes/{cand_id}")
def update_candidate(cand_id: str, payload: UpdateCandidateRequest):
    cand = next((c for c in resumes if c["id"] == cand_id), None)
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found")
    cand["name"] = payload.name
    cand["current_title"] = payload.current_title
    cand["skills"] = payload.skills
    cand["summary"] = payload.summary
    return {"status": "success", "candidate": cand}


@app.delete("/api/resumes/{cand_id}")
def delete_candidate(cand_id: str):
    global resumes
    orig_len = len(resumes)
    resumes = [c for c in resumes if c["id"] != cand_id]
    if len(resumes) == orig_len:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return {"status": "success", "remaining": len(resumes)}


@app.post("/api/jobs")
def add_job(new_job: AddJobRequest):
    global jobs, job_embeddings
    job_dict = new_job.model_dump()
    jobs.append(job_dict)
    new_text = format_job_text(job_dict)
    new_embedding = model.encode([new_text], normalize_embeddings=True)
    job_embeddings = np.vstack([job_embeddings, new_embedding])
    return {"status": "success", "total_jobs": len(jobs)}


# ==============================================================================
# USE CASE 2: AI Resume Auditor Agent (with Perfect Resume & Live Upload)
# ==============================================================================

class MissingInfoItem(BaseModel):
    category: str
    detail: str
    severity: str


class ScreeningQuestion(BaseModel):
    topic: str
    question: str
    rationale: str


class ResumeAuditReport(BaseModel):
    candidate_name: str
    detected_title: str
    completeness_score: int
    missing_critical_info: List[MissingInfoItem]
    generated_screening_questions: List[ScreeningQuestion]


RAW_UNSTRUCTURED_RESUMES = [
    {
        "id": "RAW_01",
        "name": "Marcus Vance",
        "title": "Staff Cloud & Backend Engineer",
        "missing_flaw": "Verified Perfect: Continuous employment timeline, quantifiable metrics (42%), zero gaps",
        "raw_text": """Marcus Vance - Staff Cloud & Backend Engineer
Email: marcus.vance@example.com | Phone: +1-555-0199 | San Francisco, CA

PROFESSIONAL SUMMARY:
Results-oriented engineering lead with 7+ years architecting fault-tolerant microservices and high-throughput cloud infrastructure. Reduced system latency by 42% and scaled platforms to 5M+ daily active users.

PROFESSIONAL EXPERIENCE:
Staff Cloud Engineer - CloudScale Technologies (January 2022 - Present)
- Architected Kubernetes microservices cluster processing 85,000 requests/sec with 99.99% uptime.
- Led migration of legacy monolith to AWS EKS and Terraform, reducing cloud hosting costs by 28% ($180k/yr).
- Mentored 6 mid-level software engineers and standardized CI/CD deployment pipelines using GitHub Actions.

Senior Backend Engineer - Apex Solutions (June 2018 - December 2021)
- Developed distributed payment processing engine in Go and PostgreSQL handling $14M monthly volume.
- Reduced database p99 query latency by 35% through Redis caching and PostgreSQL query optimization.
- Partnered with product and security teams to achieve SOC2 Type II compliance.

EDUCATION:
B.S. in Computer Science - University of California, Berkeley (Graduated May 2018)
GPA: 3.8 / 4.0

TECHNICAL SKILLS:
Languages: Go, Python, SQL, TypeScript
Infrastructure: AWS (EKS, RDS, S3), Docker, Kubernetes, Terraform
Databases: PostgreSQL, Redis, DynamoDB"""
    },
    {
        "id": "RAW_02",
        "name": "Priya Sharma",
        "title": "Senior Frontend Architect",
        "missing_flaw": "Employment Dates: Zero dates or tenures provided for past roles",
        "raw_text": """Priya Sharma - Senior Frontend Architect
Location: Remote

WORK HISTORY:
Lead UI Architect at TechFlow
- Re-architected entire frontend from Angular to React with Next.js.
- Supervised team of 8 frontend engineers across 3 time zones.

Frontend Developer at Digital Horizon
- Built interactive analytics dashboards using D3.js and TypeScript.
- Improved accessibility (WCAG 2.1 AA compliance).

TECHNICAL COMPETENCIES:
React, Next.js, TypeScript, Tailwind CSS, GraphQL."""
    },
    {
        "id": "RAW_03",
        "name": "Jordan Lee",
        "title": "Polymath & Tech Consultant",
        "missing_flaw": "Skill Depth: 16+ buzzwords listed without client names or impact metrics",
        "raw_text": """Jordan Lee
Self-Driven Tech Polymath & Consultant

OVERVIEW:
10+ years solving complex industry challenges across all modern software stacks. Expert in all enterprise technologies.

CORE SKILLS:
Kubernetes, React, Rust, Python, Solidity, TensorFlow, AWS, GCP, Azure, C++, Kafka, Spark, Terraform, GraphQL, iOS, Android, Cybersecurity.

PROJECTS:
- Built full-stack apps and neural networks for various clients.
- Automated CI/CD deployment pipelines and migrated databases."""
    }
]


class LocalResumeAuditorAgent:
    """
    Autonomous local agent that inspects raw unstructured resume text,
    detects real career gaps, missing employment chronologies, vague skill assertions,
    and formulates structured screening questions.
    """
    def __init__(self):
        self.name = "LocalResumeAuditorAgent-v1"

    def audit(self, text: str, name: str = "Candidate", title: str = "Candidate Profile") -> ResumeAuditReport:
        missing_info: List[MissingInfoItem] = []
        questions: List[ScreeningQuestion] = []
        score = 100

        # Extract all year mentions
        all_years = [int(y) for y in re.findall(r"\b(19\d\d|20\d\d)\b", text)]
        
        # Check 1: Missing Employment Dates / Chronology
        if not all_years or (len(all_years) <= 1 and "present" not in text.lower()):
            missing_info.append(MissingInfoItem(
                category="Employment Dates",
                detail="No chronological start or end dates provided for positions held.",
                severity="High"
            ))
            questions.append(ScreeningQuestion(
                topic="Tenure Verification",
                question="Could you provide the exact month and year durations for each of your positions?",
                rationale="Confirm verifiable years of seniority and role stability."
            ))
            score -= 35

        # Check 2: Intelligent Career Gap Detection
        # Match explicit employment ranges like (2018 - 2021) or (June 2019 - August 2021)
        range_pattern = r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+)?(20\d\d|19\d\d)\s*[-–—to]+\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+)?(20\d\d|19\d\d|Present|Current)"
        ranges_found = re.findall(range_pattern, text, re.IGNORECASE)

        parsed_ranges = []
        for start_str, end_str in ranges_found:
            start_yr = int(start_str)
            end_yr = 2026 if end_str.lower() in ["present", "current"] else int(end_str)
            parsed_ranges.append((start_yr, end_yr))

        # Sort ranges by start year
        parsed_ranges.sort(key=lambda x: x[0])

        # Check for actual gaps between consecutive job ranges
        gap_detected = False
        if len(parsed_ranges) >= 2:
            for i in range(len(parsed_ranges) - 1):
                prev_job_end = parsed_ranges[i][1]
                next_job_start = parsed_ranges[i+1][0]
                
                # A true gap occurs if next job starts 2 or more years after the previous job ended
                # e.g., ended in 2021 and next starts in 2024 (2024 - 2021 = 3 yr gap)
                actual_gap = next_job_start - prev_job_end
                if actual_gap >= 2:
                    gap_detected = True
                    missing_info.append(MissingInfoItem(
                        category="Career Gap",
                        detail=f"Detected an unexplained gap of ~{actual_gap} years between {prev_job_end} and {next_job_start}.",
                        severity="High"
                    ))
                    questions.append(ScreeningQuestion(
                        topic="Career Break Clarification",
                        question=f"We noticed an employment gap between {prev_job_end} and {next_job_start}. Could you share your activities during this period (e.g., freelance, sabbatical, or upskilling)?",
                        rationale="Ensure technical skills were actively maintained during periods between full-time roles."
                    ))
                    score -= 30
                    break

        # Check 3: Vague Skill Stuffing vs. Verified Impact
        has_metrics = bool(re.search(r"\b(\d+%\b|\$\d+|\d+\+?\s*(users|clients|engineers|microservices|apps|services|k|m))\b", text, re.IGNORECASE))
        skills_match = re.search(r"(?:SKILLS|CORE SKILLS|TECHNICAL COMPETENCIES|TECHNICAL SKILLS):?\s*(.*)", text, re.IGNORECASE | re.DOTALL)
        
        if skills_match:
            skill_blob = skills_match.group(1).split("PROJECTS:")[0].split("EDUCATION:")[0]
            skill_count = len([s for s in re.split(r"[,•\n]", skill_blob) if s.strip()])
            
            if skill_count > 10 and not has_metrics:
                missing_info.append(MissingInfoItem(
                    category="Skill Proficiency Depth",
                    detail=f"Lists {skill_count} diverse technologies without quantifiable production metrics or verified enterprise context.",
                    severity="Medium"
                ))
                questions.append(ScreeningQuestion(
                    topic="Hands-on Technical Assessment",
                    question="You list a wide variety of tools. In which two tools have you authored production code within the last 12 months?",
                    rationale="Differentiate between active production mastery and superficial familiarity."
                ))
                score -= 25

        # Check 4: Missing Measurable Impact
        if not has_metrics:
            missing_info.append(MissingInfoItem(
                category="Measurable Impact",
                detail="Work experience bullet points lack quantifiable business or technical metrics (e.g., %, $, scale).",
                severity="Low"
            ))
            questions.append(ScreeningQuestion(
                topic="Business Impact Quantification",
                question="Can you describe the scale or performance impact of the largest project you delivered?",
                rationale="Assess capability to drive tangible engineering outcomes."
            ))
            score -= 10

        # Perfect Profile Scenario (Zero Deficits Found)
        if len(missing_info) == 0:
            score = 100
            questions.append(ScreeningQuestion(
                topic="Architectural Leadership",
                question="Given your continuous track record in scaling microservices and leading infrastructure optimizations, what was the most challenging technical tradeoff you negotiated?",
                rationale="Deep-dive into architectural decision making for high-seniority verification."
            ))

        return ResumeAuditReport(
            candidate_name=name,
            detected_title=title,
            completeness_score=max(15, score),
            missing_critical_info=missing_info,
            generated_screening_questions=questions
        )

auditor_agent = LocalResumeAuditorAgent()


@app.get("/api/auditor/raw-resumes")
def get_raw_resumes():
    return RAW_UNSTRUCTURED_RESUMES


@app.post("/api/auditor/audit/{resume_id}")
def audit_raw_resume(resume_id: str):
    cand = next((r for r in RAW_UNSTRUCTURED_RESUMES if r["id"] == resume_id), None)
    if not cand:
        raise HTTPException(status_code=404, detail="Candidate not found")
    report = auditor_agent.audit(cand["raw_text"], cand["name"], cand.get("title", "Candidate"))
    return {"raw_candidate": cand, "report": report.model_dump()}


@app.post("/api/auditor/upload-audit")
async def upload_and_audit_resume(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    title: Optional[str] = Form(None)
):
    """Universal Upload for Tab 2 (Accepts PDF, DOCX, TXT, MD, RTF, JSON)."""
    contents = await file.read()
    try:
        extracted_text = extract_text_from_file_bytes(contents, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    clean_filename = re.sub(r"\.[^.]+$", "", file.filename)
    candidate_name = name.strip() if name and name.strip() else clean_filename
    candidate_title = title.strip() if title and title.strip() else "Uploaded Profile"

    report = auditor_agent.audit(extracted_text, candidate_name, candidate_title)

    custom_candidate = {
        "id": "UPLOADED_RESUME",
        "name": candidate_name,
        "title": candidate_title,
        "missing_flaw": f"Live File Audit: {file.filename}",
        "raw_text": extracted_text
    }

    return {"raw_candidate": custom_candidate, "report": report.model_dump()}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def serve_home():
    return FileResponse("static/index.html")