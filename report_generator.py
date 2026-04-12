"""
report_generator.py
Builds the LLM prompt from patient data + classification logic outputs,
calls the Anthropic API, and returns the structured markdown report.
"""

import json
from typing import Dict, List, Optional, Tuple

from literature_processing import (
    build_literature_context,
    format_curated_references_for_prompt,
    get_all_reference_titles,
)
from classification_logic import (
    classify_epilepsy_type,
    classify_seizure_type,
    score_localizations,
    get_top_localizations,
    assess_concordance,
    identify_missing_data,
)


SYSTEM_PROMPT = """You are an expert epileptologist providing a structured clinical decision-support assessment.

Your role is to analyze the provided patient data and generate a detailed, academic-quality epilepsy classification and localization report.

CRITICAL GUIDELINES:
1. Be clinically cautious and never overstate certainty when data are incomplete or conflicting.
2. Distinguish clearly between: established findings, suggestive findings, nonspecific findings, and contradictory findings.
3. Always allow "unknown" and "indeterminate" outputs when data are insufficient.
4. Do not replace clinician judgment — this is a decision-support tool.
5. Cite only references from the provided curated list. Do not invent citations.
6. When MRI and EEG disagree, explicitly state that localization is uncertain.
7. When seizure descriptions are vague, classify conservatively.
8. Use ILAE 2017 seizure and epilepsy classification terminology.
9. The tone should be professional, academic, and clinically cautious.

OUTPUT FORMAT: Return a structured markdown report with these exact sections:
## A. Brief Case Summary
## B. Proposed Epilepsy Classification
## C. Proposed Seizure Classification
## D. Proposed Localization
## E. Key Evidence Supporting the Conclusion
## F. Concordance Analysis
## G. Differential Considerations
## H. Uncertainty and Missing Data
## I. Evidence Table
## J. Literature Citations

For section I, format as a markdown table with columns: Finding | Suggests | Strength | Source (PMID) | Comment
For section J, list citations as: [PMID XXXXX] Authors. "Title." Journal (Year). DOI: ...

Strength levels: Strong / Moderate / Weak / Nonspecific
"""


def build_patient_data_summary(data: Dict) -> str:
    """Build a human-readable text summary of all entered patient data."""
    lines = []

    # Demographics
    lines.append("=== PATIENT DATA ===\n")
    lines.append(f"Age: {data.get('current_age', 'Not specified')}")
    lines.append(f"Sex: {data.get('sex', 'Not specified')}")
    lines.append(f"Handedness: {data.get('handedness', 'Not specified')}")
    lines.append(f"Age at epilepsy onset: {data.get('onset_age', 'Not specified')}")
    dur = data.get("disorder_duration")
    if dur:
        lines.append(f"Duration of disorder: {dur} years")
    lines.append("")

    # Risk factors
    rf = data.get("risk_factors", {})
    active_rf = [k.replace("_", " ").title() for k, v in rf.items() if v and k != "other_text"]
    if rf.get("other_text"):
        active_rf.append(f"Other: {rf['other_text']}")
    lines.append(f"Risk factors: {', '.join(active_rf) if active_rf else 'None reported'}")
    lines.append("")

    # Seizure burden
    lines.append(f"Seizure frequency: {data.get('seizure_frequency', 'Not specified')}")
    lines.append(f"Time since last seizure: {data.get('time_since_last_seizure', 'Not specified')}")
    lines.append(f"Drug-resistant: {data.get('drug_resistant', 'Unknown')}")
    asm = data.get("current_asms")
    if asm:
        lines.append(f"Current ASMs: {asm}")
    lines.append("")

    # Seizure semiology
    seizures = data.get("seizure_types", [])
    lines.append(f"Number of seizure types: {len(seizures)}")
    for i, sz in enumerate(seizures, 1):
        lines.append(f"\n--- Seizure Type {i} ---")
        if sz.get("seizure_description"):
            lines.append(f"Description: {sz['seizure_description']}")
        if sz.get("aura"):
            lines.append(f"Aura: YES — {sz.get('aura_description', 'described')}")
        else:
            lines.append("Aura: No")
        lines.append(f"Awareness: {sz.get('awareness', 'Unknown')}")

        features = []
        bool_features = {
            "automatisms": "Automatisms",
            "behavioral_arrest": "Behavioral arrest",
            "speech_arrest": "Speech arrest",
            "head_version": "Head version",
            "eye_deviation": "Eye deviation",
            "unilateral_tonic": "Unilateral tonic posturing",
            "clonic_movements": "Clonic movements",
            "hypermotor_features": "Hypermotor features",
            "postictal_confusion": "Postictal confusion",
            "postictal_aphasia": "Postictal aphasia",
            "todds_paresis": "Todd's paresis",
            "nocturnal": "Nocturnal predominance",
            "clustering": "Clustering",
        }
        for key, label in bool_features.items():
            if sz.get(key):
                features.append(label)
        if features:
            lines.append(f"Features: {', '.join(features)}")

        for free_feat in ["motor_features", "nonmotor_features", "sensory_symptoms",
                          "autonomic_symptoms", "emotional_symptoms", "cognitive_symptoms"]:
            val = sz.get(free_feat)
            if val:
                lines.append(f"{free_feat.replace('_', ' ').title()}: {val}")

        if sz.get("duration"):
            lines.append(f"Duration: {sz['duration']}")
        if sz.get("triggered"):
            lines.append(f"Triggered by: {sz.get('trigger_description', 'yes')}")
        if sz.get("expert_summary"):
            lines.append(f"Clinician note: {sz['expert_summary']}")

    lines.append("")

    # EEG
    eeg = data.get("eeg", {})
    if eeg.get("available"):
        lines.append("=== EEG ===")
        lines.append(f"Types performed: {', '.join(eeg.get('eeg_types', [])) or 'Not specified'}")
        lines.append(f"Result: {eeg.get('result', 'Not specified')}")
        if eeg.get("eeg_description"):
            lines.append(f"Description: {eeg['eeg_description']}")
        findings = []
        if eeg.get("focal_epileptiform"):
            findings.append(f"Focal epileptiform discharges ({eeg.get('side', '?')} {eeg.get('region', '?')})")
        if eeg.get("generalized_discharges"):
            findings.append("Generalized discharges")
        if eeg.get("focal_slowing"):
            findings.append("Focal slowing")
        if eeg.get("lateralized_abnormality"):
            findings.append(f"Lateralized abnormality ({eeg.get('side', '?')})")
        if findings:
            lines.append(f"Structured findings: {', '.join(findings)}")
        if eeg.get("ictal_onset_described") and eeg.get("ictal_eeg_description"):
            lines.append(f"Ictal EEG: {eeg['ictal_eeg_description']}")
    else:
        lines.append("EEG: Not available")

    lines.append("")

    # MRI
    mri = data.get("mri", {})
    if mri.get("available"):
        lines.append("=== MRI ===")
        lines.append(f"Result: {mri.get('result', 'Not specified')}")
        if mri.get("mri_description"):
            lines.append(f"Description: {mri['mri_description']}")
        cats = mri.get("lesion_categories", [])
        if cats:
            lines.append(f"Lesion categories: {', '.join(cats)}")
        if mri.get("side"):
            lines.append(f"Side: {mri['side']}")
        if mri.get("lobar_location"):
            lines.append(f"Lobar location: {mri['lobar_location']}")
        if mri.get("multifocal"):
            lines.append("Multifocal: Yes")
    else:
        lines.append("MRI: Not available")

    lines.append("")

    # Expert override
    expert = data.get("expert_override")
    if expert:
        lines.append(f"=== CLINICIAN IMPRESSION ===\n{expert}")

    return "\n".join(lines)


def build_llm_prompt(
    data: Dict,
    uploaded_docs: List[Dict],
) -> str:
    """
    Assemble the full LLM user prompt including:
    - Patient data summary
    - Pre-computed classification hints
    - Curated reference list
    - Uploaded literature context
    """
    # Pre-compute classification hints
    ep_type, ep_conf, ep_reasoning = classify_epilepsy_type(data)

    seizure_classifications = []
    for sz in data.get("seizure_types", []):
        sz_types = classify_seizure_type(sz)
        seizure_classifications.append(sz_types)

    loc_scores = score_localizations(data)
    top_locs = get_top_localizations(loc_scores, top_n=4)
    concordance, concordance_exp = assess_concordance(data)
    missing = identify_missing_data(data)

    # Format hints
    hint_lines = [
        "=== PRE-COMPUTED CLASSIFICATION HINTS (from structured rules) ===",
        f"Preliminary epilepsy type: {ep_type} (confidence: {ep_conf})",
        f"Rule-based reasoning: {ep_reasoning}",
        "",
        "Preliminary seizure types:",
    ]
    for i, sz_types in enumerate(seizure_classifications, 1):
        hint_lines.append(f"  Seizure {i}: {', '.join(sz_types)}")

    hint_lines.append("")
    hint_lines.append("Top localization candidates (rule-based scoring):")
    for loc, score, pmids, triggers in top_locs:
        hint_lines.append(f"  {loc} (score={score}, triggers: {'; '.join(triggers[:3])})")

    hint_lines.append(f"\nConcordance (structured): {concordance} — {concordance_exp}")

    if missing:
        hint_lines.append("\nMissing data flagged:")
        for m in missing:
            hint_lines.append(f"  - {m}")

    hints_str = "\n".join(hint_lines)

    # Patient data
    patient_str = build_patient_data_summary(data)

    # Literature context
    lit_context = build_literature_context(uploaded_docs, max_total_chars=8000)
    curated_refs = format_curated_references_for_prompt()
    all_titles = get_all_reference_titles()

    prompt = f"""Please generate a structured epilepsy classification and localization report for the following patient.

{patient_str}

{hints_str}

=== CURATED REFERENCE LIST (use these for citation) ===
{curated_refs}

=== ALL AVAILABLE REFERENCE TITLES ===
{all_titles}

"""

    if lit_context:
        prompt += f"""=== UPLOADED LITERATURE CONTENT ===
{lit_context}

"""

    prompt += """Please now generate the full structured report in markdown format following the required sections A through J.
Be clinically cautious, cite references appropriately from the lists above only, and clearly indicate confidence levels.
This report is for clinical decision-support and educational purposes only."""

    return prompt


def call_anthropic_api(prompt: str, api_key: str) -> str:
    """Call the Anthropic API and return the report text."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )

    return message.content[0].text


def generate_report(
    data: Dict,
    uploaded_docs: List[Dict],
    api_key: str,
) -> str:
    """
    Main entry point. Builds prompt, calls API, returns markdown report string.
    """
    prompt = build_llm_prompt(data, uploaded_docs)
    report_md = call_anthropic_api(prompt, api_key)
    return report_md
