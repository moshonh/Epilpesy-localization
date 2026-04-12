"""
literature_processing.py
Handles loading, parsing, and indexing of uploaded PDF/text literature.
Also provides the curated reference list from the CSV dataset.
"""

import io
import re
import pandas as pd
from typing import List, Dict, Optional

# Try optional PDF libraries
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False


# ── Curated reference list (from csv-epilepsyOR-set.csv) ──────────────────────

CURATED_REFERENCES: List[Dict] = [
    {"pmid": "24372328", "title": "Frontal lobe seizures: from clinical semiology to localization", "authors": "Bonini F et al.", "journal": "Epilepsia", "year": "2014", "doi": "10.1111/epi.12490"},
    {"pmid": "35006387", "title": "Frontal lobe seizures: overview and update", "authors": "McGonigal A.", "journal": "J Neurol", "year": "2022", "doi": "10.1007/s00415-021-10949-0"},
    {"pmid": "37137813", "title": "Multisite thalamic recordings to characterize seizure propagation", "authors": "Wu TQ et al.", "journal": "Brain", "year": "2023", "doi": "10.1093/brain/awad121"},
    {"pmid": "40217366", "title": "Absence seizures in lesion-related epilepsy", "authors": "Sun X et al.", "journal": "Acta Epileptol", "year": "2023", "doi": "10.1186/s42494-023-00133-4"},
    {"pmid": "31307620", "title": "Presurgical intracranial investigations in epilepsy surgery", "authors": "Chauvel P et al.", "journal": "Handb Clin Neurol", "year": "2019", "doi": "10.1016/B978-0-444-64142-7.00040-0"},
    {"pmid": "35964989", "title": "Semiology, EEG, and neuroimaging findings in temporal lobe epilepsies", "authors": "Frazzini V et al.", "journal": "Handb Clin Neurol", "year": "2022", "doi": "10.1016/B978-0-12-823493-8.00021-3"},
    {"pmid": "34812940", "title": "Ictal semiology of epileptic seizures with insulo-opercular genesis", "authors": "Martinez-Lizana E et al.", "journal": "J Neurol", "year": "2022", "doi": "10.1007/s00415-021-10911-0"},
    {"pmid": "30838920", "title": "The Insula and Its Epilepsies", "authors": "Jobst BC et al.", "journal": "Epilepsy Curr", "year": "2019", "doi": "10.1177/1535759718822847"},
    {"pmid": "33664202", "title": "Insular seizures and epilepsies: Ictal semiology and minimal invasive surgery", "authors": "Ryvlin P, Nguyen DK.", "journal": "Curr Opin Neurol", "year": "2021", "doi": "10.1097/WCO.0000000000000907"},
    {"pmid": "24424286", "title": "Emergence of semiology in epileptic seizures", "authors": "Chauvel P, McGonigal A.", "journal": "Epilepsy Behav", "year": "2014", "doi": "10.1016/j.yebeh.2013.12.003"},
    {"pmid": "37430420", "title": "Ictal Fear during parietal seizures", "authors": "Atacan Yaşgüçlükal M et al.", "journal": "Epileptic Disord", "year": "2023", "doi": "10.1002/epd2.20100"},
    {"pmid": "38719581", "title": "Ictal Semiology Important for Electrode Implantation and Interpretation of SEEG", "authors": "Kobayashi K, Ikeda A.", "journal": "Neurol Med Chir (Tokyo)", "year": "2024", "doi": "10.2176/jns-nmc.2023-0265"},
    {"pmid": "34052636", "title": "Epileptic aphasia - A critical appraisal", "authors": "Unterberger I et al.", "journal": "Epilepsy Behav", "year": "2021", "doi": "10.1016/j.yebeh.2021.108064"},
    {"pmid": "30974408", "title": "Localization value of ictal turning prone", "authors": "Arain AM et al.", "journal": "Seizure", "year": "2019", "doi": "10.1016/j.seizure.2018.11.003"},
    {"pmid": "36774667", "title": "Basal temporal lobe epilepsy: SEEG electroclinical characteristics", "authors": "Hadidane S et al.", "journal": "Epilepsy Res", "year": "2023", "doi": "10.1016/j.eplepsyres.2023.107090"},
    {"pmid": "34175663", "title": "Temporal lobe epilepsy: A never-ending story", "authors": "Abarrategui B et al.", "journal": "Epilepsy Behav", "year": "2021", "doi": "10.1016/j.yebeh.2021.108122"},
    {"pmid": "35906139", "title": "Visual phenomena and anatomo-electro-clinical correlations in occipital lobe seizures", "authors": "Maillard L et al.", "journal": "Rev Neurol (Paris)", "year": "2022", "doi": "10.1016/j.neurol.2022.06.001"},
    {"pmid": "36190316", "title": "'Generalized-to-focal' epilepsy: stereotactic EEG and HFO patterns", "authors": "von Ellenrieder N et al.", "journal": "Epileptic Disord", "year": "2022", "doi": "10.1684/epd.2022.1489"},
    {"pmid": "30529718", "title": "Age-dependent semiology of frontal lobe seizures", "authors": "Hintz M et al.", "journal": "Epilepsy Res", "year": "2019", "doi": "10.1016/j.eplepsyres.2018.10.007"},
    {"pmid": "26164370", "title": "Sleep-related epileptic behaviors and non-REM-related parasomnias", "authors": "Gibbs SA et al.", "journal": "Sleep Med Rev", "year": "2016", "doi": "10.1016/j.smrv.2015.05.002"},
    {"pmid": "32234986", "title": "Cingulate gyrus epilepsy: semiology, invasive EEG, and surgical approaches", "authors": "Chou CC et al.", "journal": "Neurosurg Focus", "year": "2020", "doi": "10.3171/2020.1.FOCUS19914"},
    {"pmid": "33836263", "title": "Epilepsy Classification and Terminology", "authors": "Fisher RS et al.", "journal": "Epilepsia", "year": "2017", "doi": "10.1111/epi.13670"},
    {"pmid": "34588160", "title": "Ictal fear during parietal seizures", "authors": "Yaşgüçlükal MA et al.", "journal": "Epileptic Disord", "year": "2021", "doi": "10.1684/epd.2021.1321"},
    {"pmid": "41218353", "title": "Multimodal diagnostic concordance and seizure outcomes following SEEG-guided radiofrequency thermocoagulation", "authors": "Xu Z et al.", "journal": "Epilepsy Res", "year": "2026", "doi": "10.1016/j.eplepsyres.2025.107691"},
    {"pmid": "24117237", "title": "The utility of magnetoencephalography in the presurgical evaluation of refractory insular epilepsy", "authors": "Mohamed IS et al.", "journal": "Epilepsia", "year": "2013", "doi": "10.1111/epi.12376"},
]


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a PDF file (bytes). Tries pdfplumber first, then PyPDF2."""
    text = ""
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            if text.strip():
                return text
        except Exception:
            pass

    if PYPDF2_AVAILABLE:
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            for page in reader.pages:
                text += (page.extract_text() or "") + "\n"
        except Exception:
            pass

    return text


def extract_text_from_uploaded_file(uploaded_file) -> str:
    """Extract text from a Streamlit uploaded file object (PDF or TXT)."""
    filename = uploaded_file.name.lower()
    raw_bytes = uploaded_file.read()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(raw_bytes)
    elif filename.endswith(".txt") or filename.endswith(".md"):
        try:
            return raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return raw_bytes.decode("latin-1", errors="replace")
    else:
        # Try as text
        try:
            return raw_bytes.decode("utf-8")
        except Exception:
            return ""


def chunk_text(text: str, max_chars: int = 3000, overlap: int = 200) -> List[str]:
    """Split long text into overlapping chunks."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def build_literature_context(uploaded_docs: List[Dict], max_total_chars: int = 12000) -> str:
    """
    Build a single context string from uploaded documents for inclusion
    in the LLM prompt. Truncates intelligently to fit within token budget.
    """
    if not uploaded_docs:
        return ""

    parts = []
    remaining = max_total_chars

    for doc in uploaded_docs:
        name = doc.get("name", "Unnamed")
        text = doc.get("text", "")
        if not text.strip():
            continue
        # Take first chunk of each document
        snippet = text[:min(len(text), remaining // max(len(uploaded_docs), 1))]
        parts.append(f"--- Document: {name} ---\n{snippet}")
        remaining -= len(snippet)
        if remaining <= 0:
            break

    return "\n\n".join(parts)


def format_curated_references_for_prompt(relevant_pmids: Optional[List[str]] = None) -> str:
    """
    Return a formatted string of curated references.
    If relevant_pmids provided, filter to those; otherwise return a useful subset.
    """
    refs = CURATED_REFERENCES
    if relevant_pmids:
        refs = [r for r in refs if r["pmid"] in relevant_pmids]
    
    lines = []
    for r in refs[:20]:  # limit to avoid token overflow
        lines.append(
            f"[PMID {r['pmid']}] {r['authors']} \"{r['title']}.\" {r['journal']} ({r['year']}). DOI: {r['doi']}"
        )
    return "\n".join(lines)


def get_all_reference_titles() -> str:
    """Return a compact list of all reference titles for the LLM to pick from."""
    lines = []
    for r in CURATED_REFERENCES:
        lines.append(f"PMID {r['pmid']}: {r['title']} [{r['journal']}, {r['year']}]")
    return "\n".join(lines)
