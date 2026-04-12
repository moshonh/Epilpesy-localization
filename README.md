# EpiloClassify
### Epilepsy Classification and Localization — Clinical Decision-Support Tool

---

## Overview

EpiloClassify is a Streamlit-based clinical decision-support application for epilepsy classification and localization. It accepts structured patient-level clinical data (demographics, risk factors, seizure semiology, EEG findings, MRI findings) and generates a detailed, expert-style epileptological assessment powered by the Anthropic Claude API.

The tool is grounded in a curated dataset of **340 peer-reviewed epilepsy articles** and uses a retrieval-augmented, literature-cited reasoning approach combined with structured ILAE 2017 classification rules.

> ⚠️ **DISCLAIMER:** This tool is for clinical decision-support and educational purposes only. It does not constitute a medical diagnosis and does not replace the judgment of a qualified epileptologist or neurologist.

---

## Features

- **Comprehensive patient data entry:** Demographics, risk factors, seizure burden, seizure semiology (up to 4 seizure types), EEG, MRI
- **Structured clinical reasoning:** Rule-based pre-classification using ILAE 2017 logic
- **LLM-powered expert report:** Generated via Anthropic Claude API
- **Literature-grounded citations:** 340-article curated reference set + user-uploaded PDFs
- **Structured output:** 10-section clinical report (case summary → citations)
- **Export:** Download as Markdown or plain text
- **Concordance analysis:** Automatic EEG/MRI/semiology concordance assessment

---

## Installation

### Prerequisites
- Python 3.9+
- An [Anthropic API key](https://console.anthropic.com/)

### Setup

```bash
# Clone or copy the project files
cd epilepsy_app/

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Configuration

### API Key
Set your Anthropic API key either:
- In the sidebar of the app (text input, masked)
- Via environment variable: `export ANTHROPIC_API_KEY=sk-ant-...`

---

## Project Structure

```
epilepsy_app/
├── app.py                    # Main Streamlit app (UI + session state)
├── classification_logic.py   # Structured clinical reasoning rules (ILAE 2017)
├── literature_processing.py  # PDF/text extraction + curated reference set
├── report_generator.py       # LLM prompt builder + Anthropic API caller
├── utils.py                  # Shared utilities (validation, export, sanitization)
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

### Module roles

| Module | Role |
|--------|------|
| `app.py` | Streamlit UI, form rendering, session state management |
| `classification_logic.py` | Rule-based epilepsy type / seizure type / localization scoring |
| `literature_processing.py` | PDF/text extraction, curated 340-article reference database |
| `report_generator.py` | Prompt engineering, Claude API calls, report formatting |
| `utils.py` | Validation, export, data sanitization helpers |

---

## Report Sections

The generated report includes:

| Section | Content |
|---------|---------|
| A | Brief case summary |
| B | Proposed epilepsy classification (with confidence: high/moderate/low) |
| C | Proposed seizure classification (ILAE 2017 terminology) |
| D | Proposed localization (lobe, side, network) |
| E | Key evidence (semiology / EEG / MRI / risk factors / literature) |
| F | Concordance analysis (semiology ↔ EEG ↔ MRI) |
| G | Differential considerations |
| H | Uncertainty and missing data |
| I | Evidence table (Finding / Suggests / Strength / Source / Comment) |
| J | Literature citations (curated reference set + uploaded articles) |

---

## Extending the App

The modular design allows easy extension:

- **Add new localization rules:** Edit `LOCALIZATION_RULES` in `classification_logic.py`
- **Expand reference set:** Add entries to `CURATED_REFERENCES` in `literature_processing.py`
- **Change the LLM model:** Update `model=` in `report_generator.py → call_anthropic_api()`
- **Add new output sections:** Edit `SYSTEM_PROMPT` in `report_generator.py`
- **Add PET/SPECT/MEG fields:** Extend the patient data form in `app.py` sections E/F

---

## Scientific basis

Classification logic is based on:
- ILAE 2017 operational classification of seizure types (Fisher et al., Epilepsia 2017)
- ILAE 2022 classification of the epilepsies
- Curated dataset of 340 peer-reviewed articles covering:
  - Temporal lobe epilepsy semiology and surgery
  - Frontal lobe epilepsy (including nocturnal frontal)
  - Parietal and occipital lobe epilepsy
  - Insular / insulo-opercular epilepsy
  - Cingulate gyrus epilepsy
  - Generalized epilepsy syndromes
  - EEG methodology and interpretation
  - MRI in epilepsy (FCD, MTS, lesional epilepsy)
  - Intracranial EEG / SEEG methodology

---

## License

For research and educational use. Not approved for clinical diagnosis.
