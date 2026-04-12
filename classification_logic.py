"""
classification_logic.py
Structured clinical reasoning rules for epilepsy type, seizure type,
and localization. These rules operate on the structured patient data dict
and produce preliminary classifications that feed into the LLM prompt.
"""

import re
from typing import Dict, List, Tuple, Optional


# ── Epilepsy Type ─────────────────────────────────────────────────────────────

def classify_epilepsy_type(data: Dict) -> Tuple[str, str, str]:
    """
    Returns (epilepsy_type, confidence, reasoning_note).
    confidence: 'high' | 'moderate' | 'low'
    """
    # Gather evidence
    focal_indicators = 0
    generalized_indicators = 0
    notes = []

    # MRI focal lesion
    mri = data.get("mri", {})
    if mri.get("available") and mri.get("result") == "Abnormal":
        lesion_cats = mri.get("lesion_categories", [])
        if any(c in lesion_cats for c in [
            "Mesial temporal sclerosis", "Focal cortical dysplasia",
            "Tumor", "Vascular lesion", "Post-traumatic lesion",
            "Hippocampal abnormality", "Focal cortical dysplasia"
        ]):
            focal_indicators += 3
            notes.append("MRI shows focal lesion")
        if mri.get("multifocal"):
            focal_indicators += 1
            notes.append("MRI multifocal")

    # EEG
    eeg = data.get("eeg", {})
    if eeg.get("available"):
        if eeg.get("focal_epileptiform"):
            focal_indicators += 2
            notes.append("Focal epileptiform discharges on EEG")
        if eeg.get("generalized_discharges"):
            generalized_indicators += 2
            notes.append("Generalized discharges on EEG")
        if eeg.get("focal_slowing"):
            focal_indicators += 1
            notes.append("Focal slowing on EEG")
        if eeg.get("lateralized_abnormality"):
            focal_indicators += 1

    # Seizure features - look across all seizure types
    seizures = data.get("seizure_types", [])
    for sz in seizures:
        if sz.get("awareness") in ["Impaired", "Preserved"]:
            focal_indicators += 1
        if sz.get("aura"):
            focal_indicators += 1
            notes.append("Aura present (focal indicator)")
        if sz.get("postictal_aphasia") or sz.get("todds_paresis"):
            focal_indicators += 2
            notes.append("Postictal focal deficit (focal indicator)")
        if sz.get("head_version") or sz.get("unilateral_tonic"):
            focal_indicators += 1
        if sz.get("behavioral_arrest") and not sz.get("motor_features"):
            generalized_indicators += 1  # could be absence

    # Risk factors
    rf = data.get("risk_factors", {})
    if rf.get("febrile_seizures") and not rf.get("family_history"):
        focal_indicators += 1  # FS → MTS pathway
    if rf.get("known_genetic_syndrome") or rf.get("family_history"):
        generalized_indicators += 1

    # Age at onset — some patterns
    onset_age = data.get("onset_age")
    if onset_age is not None:
        if 5 <= onset_age <= 16 and generalized_indicators >= 2:
            notes.append("Age at onset consistent with idiopathic generalized epilepsy")
        if onset_age > 20 and focal_indicators >= 2:
            notes.append("Adult-onset epilepsy favors focal etiology")

    # Decision
    if focal_indicators > 0 and generalized_indicators == 0:
        epilepsy_type = "Focal epilepsy"
        confidence = "high" if focal_indicators >= 4 else "moderate"
    elif generalized_indicators > 0 and focal_indicators == 0:
        epilepsy_type = "Generalized epilepsy"
        confidence = "high" if generalized_indicators >= 3 else "moderate"
    elif focal_indicators > 0 and generalized_indicators > 0:
        epilepsy_type = "Combined generalized and focal epilepsy"
        confidence = "moderate"
    else:
        epilepsy_type = "Unknown / insufficient data"
        confidence = "low"

    reasoning = "; ".join(notes) if notes else "Insufficient structured data for rule-based classification."
    return epilepsy_type, confidence, reasoning


# ── Seizure Type ──────────────────────────────────────────────────────────────

def classify_seizure_type(sz: Dict) -> List[str]:
    """
    For a single seizure description dict, return a list of likely seizure type labels.
    Uses ILAE 2017 classification logic.
    """
    candidates = []

    awareness = sz.get("awareness", "Unknown")
    motor = sz.get("motor_features", [])
    aura = sz.get("aura", False)
    hypermotor = sz.get("hypermotor")
    automatisms = sz.get("automatisms")
    behavioral_arrest = sz.get("behavioral_arrest")
    generalized_spread = sz.get("clonic_movements") or sz.get("unilateral_tonic")
    absence_like = behavioral_arrest and not motor and not aura

    # Focal aware seizure
    if awareness == "Preserved" and aura:
        candidates.append("Focal aware seizure")

    # Focal aware without impairment
    if awareness == "Preserved" and not aura:
        candidates.append("Focal aware seizure (motor onset)" if motor else "Focal aware seizure (non-motor onset)")

    # Focal impaired awareness
    if awareness == "Impaired":
        candidates.append("Focal impaired awareness seizure")

    # Focal to bilateral tonic-clonic
    if sz.get("postictal_confusion") or sz.get("todds_paresis"):
        candidates.append("Focal to bilateral tonic-clonic seizure")
    elif generalized_spread and awareness == "Impaired":
        candidates.append("Focal to bilateral tonic-clonic seizure (possible)")

    # Absence
    if absence_like and not aura:
        candidates.append("Absence seizure (possible)")

    # Hypermotor seizure (frontal)
    if hypermotor:
        candidates.append("Hypermotor (frontal) seizure")

    # Tonic seizure
    if sz.get("unilateral_tonic") and not automatisms:
        candidates.append("Tonic seizure")

    if not candidates:
        candidates.append("Seizure type undetermined — insufficient detail")

    return candidates


# ── Localization ──────────────────────────────────────────────────────────────

LOCALIZATION_RULES = [
    # (description, localizations, score_weight, pmids)
    # Temporal
    ("Déjà vu / jamais vu aura",
     ["Mesial temporal (likely left dominant)"], 3,
     ["35964989", "34175663"]),
    ("Epigastric rising aura",
     ["Mesial temporal lobe"], 3,
     ["35964989"]),
    ("Oral automatisms + impaired awareness",
     ["Temporal lobe (mesial)"], 3,
     ["35964989", "34175663"]),
    ("Behavioral arrest + oroalimentary automatisms + postictal confusion",
     ["Mesial temporal lobe (MTLE pattern)"], 3,
     ["35964989"]),
    ("Auditory aura (simple tones)",
     ["Superior temporal gyrus / lateral temporal"], 2,
     ["35964989"]),
    ("Language disturbance postictal",
     ["Left temporal lobe"], 2,
     ["34052636"]),
    ("Postictal aphasia",
     ["Left hemisphere (language-dominant temporal)"], 3,
     ["34052636"]),
    # Frontal
    ("Hypermotor features",
     ["Frontal lobe (SEEG-confirmed pattern)"], 3,
     ["24372328", "35006387"]),
    ("Nocturnal predominance with hypermotor behavior",
     ["Frontal lobe (nocturnal frontal lobe epilepsy pattern)"], 3,
     ["26164370", "35006387"]),
    ("Short seizures (<30 sec) with rapid recovery",
     ["Frontal lobe"], 2,
     ["24372328"]),
    ("Head version or eye deviation (contralateral)",
     ["Frontal lobe (FEF) or parietal"], 2,
     ["24372328"]),
    ("Speech arrest (ictal)",
     ["Frontal (Broca area adjacent) or language-dominant temporal"], 2,
     ["34052636"]),
    ("Vocalization / ictal cry",
     ["Supplementary motor area (SMA) or frontal"], 2,
     ["24372328"]),
    ("Bilateral asymmetric tonic posturing",
     ["SMA / mesial frontal"], 3,
     ["35006387"]),
    # Parietal
    ("Somatosensory aura (tingling, numbness)",
     ["Parietal lobe (somatosensory cortex)"], 3,
     ["37430420", "34588160"]),
    ("Ictal fear (parietal pattern)",
     ["Parietal lobe (cingulate or amygdala projection)"], 2,
     ["37430420"]),
    # Occipital
    ("Visual aura (phosphenes, colors)",
     ["Occipital lobe"], 3,
     ["35906139"]),
    ("Visual hallucinations (simple)",
     ["Occipital lobe / visual cortex"], 3,
     ["35906139"]),
    ("Eye deviation early in seizure + visual aura",
     ["Occipital lobe (posterior)"], 2,
     ["35906139"]),
    # Insular
    ("Laryngeal / throat sensation aura",
     ["Insular cortex"], 3,
     ["34812940", "30838920", "33664202"]),
    ("Autonomic features prominent (heart racing, sweating)",
     ["Insular cortex (autonomic representation)"], 2,
     ["30838920"]),
    ("Painful aura",
     ["Insular cortex or parietal"], 2,
     ["34812940"]),
    # Cingulate
    ("Emotional aura (fear, anxiety) + hypermotor",
     ["Anterior cingulate / cingulate epilepsy"], 2,
     ["32234986"]),
]


def score_localizations(data: Dict) -> Dict[str, Dict]:
    """
    Score possible localizations based on structured data.
    Returns dict: {localization_label: {'score': int, 'pmids': [...], 'triggers': [...]}}
    """
    scores = {}

    # Collect all relevant text and features from seizure types
    all_features = set()
    all_text = ""

    seizures = data.get("seizure_types", [])
    for sz in seizures:
        if sz.get("aura") and sz.get("aura_description"):
            all_text += sz["aura_description"].lower() + " "
        if sz.get("seizure_description"):
            all_text += sz["seizure_description"].lower() + " "
        # Boolean features
        for feat in ["hypermotor_features", "behavioral_arrest", "automatisms",
                     "head_version", "eye_deviation", "speech_arrest",
                     "postictal_aphasia", "todds_paresis", "nocturnal"]:
            if sz.get(feat):
                all_features.add(feat)
        if sz.get("autonomic_symptoms"):
            all_features.add("autonomic")
        if sz.get("emotional_symptoms"):
            all_features.add("emotional")
        if sz.get("sensory_symptoms"):
            all_features.add("sensory")

    # EEG
    eeg = data.get("eeg", {})
    eeg_side = eeg.get("side", "")
    eeg_region = eeg.get("region", "")
    eeg_text = (eeg.get("eeg_description", "") or "").lower()
    if eeg.get("ictal_onset_described") and eeg.get("ictal_eeg_description"):
        eeg_text += " " + eeg["ictal_eeg_description"].lower()

    # MRI
    mri = data.get("mri", {})
    mri_side = mri.get("side", "")
    mri_lobe = mri.get("lobar_location", "")
    mri_lesion_cats = mri.get("lesion_categories", [])

    # Strong MRI anchor
    if mri.get("available") and mri.get("result") == "Abnormal":
        if "Mesial temporal sclerosis" in mri_lesion_cats or "Hippocampal abnormality" in mri_lesion_cats:
            loc = f"Mesial temporal lobe ({mri_side or 'unspecified side'})"
            scores[loc] = scores.get(loc, {"score": 0, "pmids": [], "triggers": []})
            scores[loc]["score"] += 5
            scores[loc]["pmids"].extend(["35964989", "34175663"])
            scores[loc]["triggers"].append("MRI: mesial temporal sclerosis / hippocampal abnormality")
        if "Focal cortical dysplasia" in mri_lesion_cats:
            lobe_str = mri_lobe or "unknown lobe"
            loc = f"Focal cortical dysplasia region ({mri_side or '?'} {lobe_str})"
            scores[loc] = scores.get(loc, {"score": 0, "pmids": [], "triggers": []})
            scores[loc]["score"] += 4
            scores[loc]["pmids"].extend(["31307620"])
            scores[loc]["triggers"].append("MRI: focal cortical dysplasia")
        if mri_lobe:
            loc = f"{mri_lobe} lobe ({mri_side or 'unspecified side'}) — MRI lesion"
            scores[loc] = scores.get(loc, {"score": 0, "pmids": [], "triggers": []})
            scores[loc]["score"] += 3
            scores[loc]["triggers"].append(f"MRI lobar location: {mri_lobe}")

    # EEG anchor
    if eeg.get("available") and eeg.get("focal_epileptiform") and eeg_region:
        loc = f"{eeg_region} ({eeg_side or 'unspecified side'}) — EEG-based"
        scores[loc] = scores.get(loc, {"score": 0, "pmids": [], "triggers": []})
        scores[loc]["score"] += 3
        scores[loc]["triggers"].append(f"EEG: focal epileptiform in {eeg_region} ({eeg_side})")

    # Text/feature-based rules
    for (description, locs, weight, pmids) in LOCALIZATION_RULES:
        triggered = False
        desc_lower = description.lower()

        # Check free text
        keywords = re.findall(r'\b\w{4,}\b', desc_lower)
        match_count = sum(1 for kw in keywords if kw in all_text)
        if match_count >= max(1, len(keywords) // 3):
            triggered = True

        # Check structured features
        if "hypermotor" in desc_lower and "hypermotor_features" in all_features:
            triggered = True
        if "nocturnal" in desc_lower and "nocturnal" in all_features:
            triggered = True
        if "speech arrest" in desc_lower and "speech_arrest" in all_features:
            triggered = True
        if "postictal aphasia" in desc_lower and "postictal_aphasia" in all_features:
            triggered = True
        if "todd" in desc_lower and "todds_paresis" in all_features:
            triggered = True
        if "head version" in desc_lower and "head_version" in all_features:
            triggered = True
        if "autonomic" in desc_lower and "autonomic" in all_features:
            triggered = True
        if "emotional" in desc_lower and "emotional" in all_features:
            triggered = True
        if "somatosensory" in desc_lower and "sensory" in all_features:
            triggered = True

        if triggered:
            for loc in locs:
                scores[loc] = scores.get(loc, {"score": 0, "pmids": [], "triggers": []})
                scores[loc]["score"] += weight
                scores[loc]["pmids"].extend(pmids)
                scores[loc]["triggers"].append(description)

    return scores


def get_top_localizations(scores: Dict, top_n: int = 3) -> List[Tuple[str, int, List[str]]]:
    """Return top N localizations sorted by score."""
    sorted_locs = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
    result = []
    for loc, v in sorted_locs[:top_n]:
        unique_pmids = list(dict.fromkeys(v["pmids"]))
        result.append((loc, v["score"], unique_pmids, v["triggers"]))
    return result


def assess_concordance(data: Dict) -> Tuple[str, str]:
    """
    Assess concordance between semiology, EEG, and MRI.
    Returns (concordance_label, explanation).
    """
    eeg = data.get("eeg", {})
    mri = data.get("mri", {})

    has_eeg = eeg.get("available", False)
    has_mri = mri.get("available", False)

    if not has_eeg and not has_mri:
        return "Cannot assess", "Neither EEG nor MRI data are available."

    eeg_side = eeg.get("side", "").lower()
    eeg_region = (eeg.get("region", "") or "").lower()
    mri_side = (mri.get("side", "") or "").lower()
    mri_lobe = (mri.get("lobar_location", "") or "").lower()

    if has_eeg and has_mri:
        if eeg_side and mri_side and eeg_side != mri_side:
            return "Discordant", (
                f"EEG lateralization ({eeg_side}) and MRI lesion side ({mri_side}) are on opposite sides. "
                "This discordance limits localization certainty significantly."
            )
        if eeg_region and mri_lobe and eeg_region not in mri_lobe and mri_lobe not in eeg_region:
            return "Partial concordance", (
                f"EEG region ({eeg_region}) and MRI lobar location ({mri_lobe}) are not clearly concordant. "
                "Further investigation (SEEG/icEEG) may be needed."
            )
        if (eeg_side == mri_side or not (eeg_side and mri_side)):
            if has_eeg and has_mri:
                return "Strong concordance", (
                    "EEG and MRI findings are concordant in lateralization and/or region."
                )

    if has_eeg and not has_mri:
        return "Partial (EEG only)", "MRI data unavailable; concordance based on EEG and semiology only."

    if has_mri and not has_eeg:
        return "Partial (MRI only)", "EEG data unavailable; concordance based on MRI and semiology only."

    return "Indeterminate", "Insufficient data for concordance assessment."


def identify_missing_data(data: Dict) -> List[str]:
    """Return a list of important missing data items."""
    missing = []
    if not data.get("eeg", {}).get("available"):
        missing.append("No EEG data available — critical for classification")
    elif not data.get("eeg", {}).get("ictal_onset_described"):
        missing.append("No ictal EEG described — important for onset localization")

    if not data.get("mri", {}).get("available"):
        missing.append("No MRI data available — important for structural etiology")
    elif data.get("mri", {}).get("result") == "Indeterminate":
        missing.append("MRI result is indeterminate — consider 3T MRI with epilepsy protocol")

    seizures = data.get("seizure_types", [])
    if not seizures:
        missing.append("No seizure semiology data entered")
    for i, sz in enumerate(seizures):
        if not sz.get("seizure_description") and not sz.get("aura"):
            missing.append(f"Seizure type {i+1}: no semiology description provided")

    rf = data.get("risk_factors", {})
    if not any(rf.values()):
        missing.append("No risk factors identified or entered")

    if not data.get("current_age"):
        missing.append("Patient age missing")
    if not data.get("onset_age"):
        missing.append("Age at epilepsy onset missing")

    return missing
