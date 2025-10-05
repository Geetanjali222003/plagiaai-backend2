from fastapi import FastAPI, File, UploadFile
import fitz  # PyMuPDF for PDF
import docx
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

app = FastAPI(title="Mini Turnitin")

# Load sources.json (URLs to check)
with open("sources.json", "r", encoding="utf-8") as f:
    SOURCES = json.load(f)["urls"]

def extract_text_from_pdf(file_path):
    text = ""
    pdf = fitz.open(file_path)
    for page in pdf:
        text += page.get_text()
    return text.strip()

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])

def fetch_text_from_url(url):
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return ""
        soup = BeautifulSoup(resp.text, "html.parser")
        return " ".join([p.get_text() for p in soup.find_all("p")])
    except:
        return ""

@app.post("/check_plagiarism/")
async def check_plagiarism(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    file_location = f"temp_{file.filename}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Extract text from PDF/DOCX
    if file.filename.endswith(".pdf"):
        input_text = extract_text_from_pdf(file_location)
    elif file.filename.endswith(".docx"):
        input_text = extract_text_from_docx(file_location)
    else:
        return {"error": "Only PDF and DOCX are supported"}

    if not input_text.strip():
        return {"error": "No text found in file"}

    # Collect website texts
    matches = []
    for url in SOURCES:
        site_text = fetch_text_from_url(url)
        if not site_text.strip():
            continue
        vectorizer = TfidfVectorizer().fit([input_text, site_text])
        tfidf_matrix = vectorizer.transform([input_text, site_text])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        if sim > 0.2:  # threshold
            matches.append({"url": url, "similarity": round(sim * 100, 2)})

    matches = sorted(matches, key=lambda x: x["similarity"], reverse=True)[:5]  # top 5

    plagiarism_percent = max([m["similarity"] for m in matches], default=0)

    return {
        "plagiarism_percent": plagiarism_percent,
        "matches": matches,
        "summary": input_text[:500] + "..."
    }
