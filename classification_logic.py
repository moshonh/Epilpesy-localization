"""
classification_logic.py  v2
Semiology-first localization. Full free-text parsing.
Dystonia, automatisms, and all motor signs explicitly detected.
"""

import re
from typing import Dict, List, Tuple, Optional


def extract_all_semiology_text(data: Dict) -> str:
    parts = []
    for sz in data.get("seizure_types", []):
        for field in ["seizure_description","aura_description","motor_features",
                      "nonmotor_features","sensory_symptoms","autonomic_symptoms",
                      "emotional_symptoms","cognitive_symptoms","trigger_description","expert_summary"]:
            val = sz.get(field) or ""
            if val:
                parts.append(val.lower())
    return " ".join(parts)


def extract_all_boolean_features(data: Dict) -> set:
    features = set()
    for sz in data.get("seizure_types", []):
        for feat in ["aura","automatisms","behavioral_arrest","speech_arrest","head_version",
                     "eye_deviation","unilateral_tonic","clonic_movements","hypermotor_features",
                     "postictal_confusion","postictal_aphasia","todds_paresis","nocturnal","clustering"]:
            if sz.get(feat):
                features.add(feat)
        for feat in ["autonomic_symptoms","emotional_symptoms","sensory_symptoms","motor_features","cognitive_symptoms"]:
            if sz.get(feat):
                features.add(feat)
    return features


def text_contains(text: str, keywords: List[str]) -> bool:
    return any(kw in text for kw in keywords)


SEMIOLOGY_KEYWORD_RULES = [
    # MESIAL TEMPORAL
    {"localization":"Mesial temporal lobe","score":4,"pmids":["35964989","34175663"],
     "label":"Epigastric / rising aura",
     "keywords":["epigastric","rising","nausea","gastric","abdominal","stomach"]},
    {"localization":"Mesial temporal lobe","score":4,"pmids":["35964989","34175663"],
     "label":"Deja vu / jamais vu",
     "keywords":["deja vu","déjà vu","jamais vu","familiarity","deja vecu","already seen"]},
    {"localization":"Mesial temporal lobe","score":4,"pmids":["35964989","34175663"],
     "label":"Oroalimentary automatisms (lip smacking, chewing, swallowing)",
     "keywords":["lip smacking","chewing","swallowing","oral automatism","oroalimentary",
                 "oro-alimentary","licking","gulping","lip"]},
    {"localization":"Mesial temporal lobe","score":3,"pmids":["35964989","34175663"],
     "label":"Manual / gestural automatisms",
     "keywords":["hand automatism","manual automatism","gestural","fumbling","picking",
                 "rubbing hands","patting","fidgeting"]},
    {"localization":"Mesial temporal lobe","score":3,"pmids":["35964989","34175663"],
     "label":"Behavioral arrest / staring",
     "keywords":["behavioral arrest","staring spell","blank stare","unresponsive",
                 "motionless","freezing","blank","stare"]},
    {"localization":"Mesial temporal lobe","score":3,"pmids":["35964989"],
     "label":"Olfactory / gustatory aura",
     "keywords":["smell","olfactory","gustatory","taste","odor","unpleasant smell"]},
    {"localization":"Mesial temporal lobe (amygdala / fear network)","score":3,
     "pmids":["34588160","12600809"],
     "label":"Ictal fear / anxiety",
     "keywords":["fear","terror","anxiety","panic","dread","frightened","scared","ictal fear"]},
    {"localization":"Mesial temporal lobe","score":2,"pmids":["35964989"],
     "label":"Postictal confusion",
     "keywords":["postictal confusion","postictal disorientation","confused after",
                 "slow recovery","prolonged recovery","confusion after"]},
    # DYSTONIA — key lateralizing sign
    {"localization":"Temporal or frontal lobe — CONTRALATERAL to dystonic limb (strong lateralizing sign)",
     "score":4,"pmids":["35964989","34175663"],
     "label":"Dystonic posturing (contralateral to seizure onset — highly lateralizing)",
     "keywords":["dystonia","dystonic","dystonic posturing","dystonic arm","tonic dystonic",
                 "unilateral dystonia","arm dystonia","contralateral dystonia","dystonic limb"]},
    # TODD'S
    {"localization":"Contralateral hemisphere — Todd's paresis (strong lateralizing sign)",
     "score":4,"pmids":["35964989"],
     "label":"Todd's paresis",
     "keywords":["todd","todd's paresis","postictal paresis","postictal weakness",
                 "hemiparesis","postictal hemiplegia"]},
    # LATERAL TEMPORAL
    {"localization":"Lateral (neocortical) temporal lobe","score":3,"pmids":["35964989"],
     "label":"Auditory aura",
     "keywords":["auditory","hearing","buzzing","ringing","tinnitus","sound","voices","music"]},
    {"localization":"Left temporal / language cortex","score":3,"pmids":["34052636"],
     "label":"Aphasia (ictal or postictal)",
     "keywords":["aphasia","dysphasia","language","word finding","postictal aphasia",
                 "cannot speak","unable to speak","ictal aphasia","speech disturbance"]},
    {"localization":"Basal temporal lobe","score":3,"pmids":["36774667"],
     "label":"Complex / formed visual hallucinations",
     "keywords":["formed visual","faces","scenes","complex visual","people appearing"]},
    # FRONTAL
    {"localization":"Frontal lobe (hypermotor / SMA)","score":4,"pmids":["24372328","35006387"],
     "label":"Hypermotor seizure",
     "keywords":["hypermotor","thrashing","cycling","kicking","bicycling","rocking",
                 "pelvic","agitated","violent movement","flailing","motor agitation"]},
    {"localization":"Frontal lobe (SMA / mesial frontal — fencing / M2e posture)","score":4,
     "pmids":["35006387"],
     "label":"Bilateral asymmetric tonic posturing (fencing posture)",
     "keywords":["fencing","asymmetric tonic","m2e","figure-of-4","figure of 4",
                 "arm extension","supplementary motor","sma","mesial frontal"]},
    {"localization":"Frontal lobe (NFLE — nocturnal frontal lobe epilepsy)","score":4,
     "pmids":["26164370","35006387"],
     "label":"Nocturnal seizures from sleep",
     "keywords":["nocturnal","from sleep","night","awakening from sleep","nrem","non-rem",
                 "sleep seizure","wake from sleep"]},
    {"localization":"Frontal lobe (Broca / opercular)","score":3,"pmids":["24372328","34052636"],
     "label":"Vocalization / ictal cry / speech arrest",
     "keywords":["vocalization","ictal cry","grunting","speech arrest","moan","screaming","yell"]},
    {"localization":"Frontal lobe (FEF / premotor)","score":3,"pmids":["24372328"],
     "label":"Head version / forced eye deviation",
     "keywords":["head version","head turning","eye deviation","forced gaze","versive",
                 "contralateral gaze","head turn"]},
    {"localization":"Frontal lobe (cingulate)","score":3,"pmids":["32234986"],
     "label":"Cingulate pattern",
     "keywords":["cingulate","anterior cingulate","cingulate gyrus"]},
    # PARIETAL
    {"localization":"Parietal lobe (somatosensory cortex)","score":4,
     "pmids":["37430420","34588160"],
     "label":"Somatosensory aura (tingling, numbness, electric)",
     "keywords":["tingling","numbness","pins and needles","electric","somatosensory",
                 "paresthesia","paraesthesia","buzzing sensation","crawling sensation","shock"]},
    {"localization":"Posterior parietal / precuneus","score":3,"pmids":["26235442"],
     "label":"Visuospatial / body schema symptoms",
     "keywords":["visuospatial","spatial disorientation","body schema","precuneus",
                 "out of body","depersonalization","derealization"]},
    # OCCIPITAL
    {"localization":"Occipital lobe (primary visual cortex)","score":4,"pmids":["35906139"],
     "label":"Elementary visual aura (phosphenes, colors, flashing)",
     "keywords":["phosphene","phosphenes","visual aura","flashing light","colors","coloured",
                 "colored","scintillating","zigzag","fortification","elementary visual",
                 "simple visual","sparks","lights"]},
    {"localization":"Occipital lobe","score":3,"pmids":["35906139"],
     "label":"Ictal blindness / amaurosis",
     "keywords":["blindness","amaurosis","visual loss","cannot see","ictal blindness"]},
    # INSULAR
    {"localization":"Insular cortex (highly specific)","score":4,
     "pmids":["34812940","30838920","33664202"],
     "label":"Laryngeal / throat / choking sensation",
     "keywords":["laryngeal","throat","choking","suffocating","suffocation","strangling",
                 "constriction in throat","lump in throat","larynx","pharyngeal","chest tightness"]},
    {"localization":"Insular cortex (autonomic)","score":3,"pmids":["30838920","33664202"],
     "label":"Prominent autonomic aura (cardiac, sweating, piloerection)",
     "keywords":["heart racing","palpitation","tachycardia","sweating","piloerection",
                 "goosebumps","flushing","breathless","hot feeling","cold feeling"]},
    {"localization":"Insular cortex (pain)","score":3,"pmids":["34812940"],
     "label":"Painful aura",
     "keywords":["pain","painful","burning","electric shock","painful sensation","ictal pain"]},
    {"localization":"Insular / opercular cortex","score":3,"pmids":["34812940","30838920"],
     "label":"Perioral / facial somatosensory symptoms",
     "keywords":["perioral","facial tingling","jaw","tongue","face tingling",
                 "facial twitching","opercular"]},
]

BOOLEAN_LOCALIZATION_MAP = {
    "automatisms":[("Mesial temporal lobe",3,["35964989","34175663"],"Automatisms (checkbox)")],
    "behavioral_arrest":[("Mesial temporal lobe",2,["35964989"],"Behavioral arrest (checkbox)")],
    "hypermotor_features":[("Frontal lobe (hypermotor pattern)",4,["24372328","35006387"],"Hypermotor features (checkbox)")],
    "postictal_aphasia":[("Left hemisphere — language-dominant",4,["34052636"],"Postictal aphasia (checkbox)")],
    "todds_paresis":[("Contralateral hemisphere — Todd's paresis",4,["35964989"],"Todd's paresis (checkbox)")],
    "head_version":[("Frontal lobe (FEF) — contralateral to version",3,["24372328"],"Head version (checkbox)")],
    "eye_deviation":[("Frontal lobe (FEF) — contralateral",2,["24372328"],"Eye deviation (checkbox)")],
    "nocturnal":[("Frontal lobe (NFLE)",3,["26164370","35006387"],"Nocturnal predominance (checkbox)")],
    "speech_arrest":[("Frontal / left temporal — language cortex",3,["34052636","24372328"],"Speech arrest (checkbox)")],
    "unilateral_tonic":[("Contralateral frontal or temporal lobe",3,["24372328"],"Unilateral tonic posturing (checkbox)")],
    "sensory_symptoms":[("Parietal lobe (somatosensory)",2,["37430420"],"Sensory symptoms (checkbox)")],
    "autonomic_symptoms":[("Insular cortex or mesial temporal",2,["30838920"],"Autonomic symptoms (checkbox)")],
}


def score_localizations(data: Dict) -> Dict[str, Dict]:
    scores: Dict[str, Dict] = {}

    def add_score(loc, pts, pmids, trigger):
        if loc not in scores:
            scores[loc] = {"score":0,"pmids":[],"triggers":[]}
        scores[loc]["score"] += pts
        scores[loc]["pmids"].extend(pmids)
        if trigger not in scores[loc]["triggers"]:
            scores[loc]["triggers"].append(trigger)

    # 1. FREE TEXT (primary — highest weight)
    all_text = extract_all_semiology_text(data)
    for rule in SEMIOLOGY_KEYWORD_RULES:
        if text_contains(all_text, rule["keywords"]):
            add_score(rule["localization"], rule["score"], rule["pmids"], rule["label"])

    # 2. BOOLEAN FEATURES (primary)
    bool_features = extract_all_boolean_features(data)
    for feat, entries in BOOLEAN_LOCALIZATION_MAP.items():
        if feat in bool_features:
            for (loc, pts, pmids, label) in entries:
                add_score(loc, pts, pmids, label)

    # 3. EEG (confirmatory — secondary)
    eeg = data.get("eeg", {})
    if eeg.get("available"):
        eeg_side = eeg.get("side","")
        eeg_region = eeg.get("region","")
        if eeg.get("focal_epileptiform") and eeg_region and eeg_region != "Not specified":
            add_score(f"{eeg_region} ({eeg_side}) — EEG focal epileptiform",
                      2, [], f"EEG: focal epileptiform {eeg_region} ({eeg_side})")
        if eeg.get("ictal_onset_described") and eeg_region and eeg_region != "Not specified":
            add_score(f"{eeg_region} ({eeg_side}) — Ictal EEG onset",
                      3, ["31307620"], f"Ictal EEG onset: {eeg_region} ({eeg_side})")
        if eeg.get("focal_slowing") and eeg_region and eeg_region != "Not specified":
            add_score(f"{eeg_region} ({eeg_side}) — EEG focal slowing",
                      1, [], f"EEG: focal slowing {eeg_region}")

    # 4. MRI (confirmatory — secondary)
    mri = data.get("mri", {})
    if mri.get("available") and mri.get("result") == "Abnormal":
        mri_side = mri.get("side","")
        mri_lobe = mri.get("lobar_location","")
        cats = mri.get("lesion_categories",[])
        if any(c in cats for c in ["Mesial temporal sclerosis","Hippocampal abnormality"]):
            add_score(f"Mesial temporal lobe ({mri_side}) — MRI: MTS",
                      3,["35964989","34175663"],"MRI: mesial temporal sclerosis")
        if "Focal cortical dysplasia" in cats:
            add_score(f"FCD — {mri_side} {mri_lobe}",
                      3,["31307620"],"MRI: focal cortical dysplasia")
        if mri_lobe and mri_lobe != "Not specified":
            add_score(f"{mri_lobe} lobe ({mri_side}) — MRI structural lesion",
                      2,[],"MRI structural lesion")
    return scores


def get_top_localizations(scores: Dict, top_n: int = 4) -> List[Tuple]:
    sorted_locs = sorted(scores.items(), key=lambda x: x[1]["score"], reverse=True)
    return [(loc, v["score"], list(dict.fromkeys(v["pmids"])), v["triggers"])
            for loc, v in sorted_locs[:top_n]]


def classify_epilepsy_type(data: Dict) -> Tuple[str, str, str]:
    focal_score = 0
    generalized_score = 0
    notes = []
    all_text = extract_all_semiology_text(data)
    bool_features = extract_all_boolean_features(data)

    focal_kws = ["epigastric","deja vu","déjà vu","tingling","visual aura","laryngeal",
                 "throat","dystonia","dystonic","lip smacking","chewing","hypermotor",
                 "versive","head version","aura","automatism"]
    if any(kw in all_text for kw in focal_kws):
        focal_score += 2
        notes.append("Semiological keywords suggest focal onset")

    for feat, pts, note in [
        ("aura",2,"Aura present"),("automatisms",2,"Automatisms"),
        ("postictal_aphasia",3,"Postictal aphasia"),("todds_paresis",3,"Todd's paresis"),
        ("hypermotor_features",2,"Hypermotor features"),("unilateral_tonic",2,"Unilateral tonic posturing"),
    ]:
        if feat in bool_features:
            focal_score += pts
            notes.append(note)

    gen_kws = ["absence","myoclonic","myoclonus","juvenile","generalized tonic-clonic",
               "drop attack","atonic","generalized"]
    if any(kw in all_text for kw in gen_kws):
        generalized_score += 2
        notes.append("Semiological keywords suggest generalized onset")

    eeg = data.get("eeg", {})
    if eeg.get("available"):
        if eeg.get("focal_epileptiform"):
            focal_score += 2; notes.append("EEG: focal epileptiform")
        if eeg.get("generalized_discharges"):
            generalized_score += 2; notes.append("EEG: generalized discharges")
        if eeg.get("focal_slowing"):
            focal_score += 1

    mri = data.get("mri", {})
    if mri.get("available") and mri.get("result") == "Abnormal":
        cats = mri.get("lesion_categories", [])
        if any(c in cats for c in ["Mesial temporal sclerosis","Focal cortical dysplasia",
                                    "Tumor","Vascular lesion","Hippocampal abnormality"]):
            focal_score += 2; notes.append("MRI: focal structural lesion")

    rf = data.get("risk_factors", {})
    if rf.get("febrile_seizures"):
        focal_score += 1; notes.append("Risk: febrile seizures")
    if rf.get("known_genetic_syndrome") or rf.get("family_history"):
        generalized_score += 1

    if focal_score > 0 and generalized_score == 0:
        return "Focal epilepsy", ("high" if focal_score >= 6 else "moderate"), "; ".join(notes)
    elif generalized_score > 0 and focal_score == 0:
        return "Generalized epilepsy", ("high" if generalized_score >= 4 else "moderate"), "; ".join(notes)
    elif focal_score > 0 and generalized_score > 0:
        return "Combined generalized and focal epilepsy", "moderate", "; ".join(notes)
    else:
        return "Unknown / insufficient data", "low", "Insufficient data."


def classify_seizure_type(sz: Dict) -> List[str]:
    candidates = []
    awareness = sz.get("awareness","Unknown")
    aura = sz.get("aura", False)
    automatisms = sz.get("automatisms", False)
    behavioral_arrest = sz.get("behavioral_arrest", False)
    hypermotor = sz.get("hypermotor_features", False)
    unilateral_tonic = sz.get("unilateral_tonic", False)
    clonic = sz.get("clonic_movements", False)
    postictal_confusion = sz.get("postictal_confusion", False)
    postictal_aphasia = sz.get("postictal_aphasia", False)
    todds = sz.get("todds_paresis", False)

    text_parts = []
    for f in ["seizure_description","aura_description","motor_features","nonmotor_features",
              "sensory_symptoms","autonomic_symptoms","emotional_symptoms","cognitive_symptoms"]:
        val = sz.get(f) or ""
        if val: text_parts.append(val.lower())
    sz_text = " ".join(text_parts)

    has_dystonia = text_contains(sz_text, ["dystonia","dystonic","dystonic posturing","arm dystonia"])
    has_hypermotor_text = text_contains(sz_text, ["hypermotor","thrashing","cycling","kicking","bicycling","rocking","agitated","flailing"])
    has_gtcs = text_contains(sz_text, ["tonic-clonic","grand mal","bilateral tonic","generalized convulsion","convulsion"])
    has_myoclonic = text_contains(sz_text, ["myoclonic","myoclonus","jerk","jerking"])
    has_oral_auto = text_contains(sz_text, ["lip","chew","swallow","oral","lick","gulp"])
    has_manual_auto = text_contains(sz_text, ["hand","fumble","pick","rub","pat","gestural","fidget"])

    if awareness == "Preserved" and (aura or sz.get("sensory_symptoms") or sz.get("motor_features")):
        candidates.append("Focal aware seizure (FAS)")
    if awareness == "Impaired":
        candidates.append("Focal impaired awareness seizure (FIAS)")
    if todds or postictal_aphasia or has_gtcs or (postictal_confusion and (unilateral_tonic or clonic)):
        candidates.append("Focal to bilateral tonic-clonic seizure (FBTCS)")
    if hypermotor or has_hypermotor_text:
        candidates.append("Hypermotor seizure (frontal lobe pattern)")
    if has_dystonia:
        candidates.append("Seizure with dystonic posturing — LATERALIZING SIGN (onset contralateral to dystonia)")
    if (unilateral_tonic or text_contains(sz_text, ["tonic","stiffening","rigid"])) and not has_gtcs:
        candidates.append("Tonic seizure / tonic component")
    if has_myoclonic:
        candidates.append("Myoclonic component")
    if automatisms:
        if has_oral_auto:
            candidates.append("Seizure with oroalimentary automatisms (temporal lobe pattern)")
        elif has_manual_auto:
            candidates.append("Seizure with manual automatisms (temporal lobe pattern)")
        else:
            candidates.append("Seizure with automatisms")
    if behavioral_arrest and not aura and not automatisms and awareness == "Unknown":
        candidates.append("Possible absence seizure — consider focal vs generalized")

    if not candidates:
        candidates.append("Seizure type undetermined — insufficient detail")
    return list(dict.fromkeys(candidates))


def assess_concordance(data: Dict) -> Tuple[str, str]:
    eeg = data.get("eeg", {})
    mri = data.get("mri", {})
    has_eeg = eeg.get("available", False)
    has_mri = mri.get("available", False)
    if not has_eeg and not has_mri:
        return "Cannot assess", "Neither EEG nor MRI available."
    eeg_side = (eeg.get("side") or "").lower().strip()
    mri_side = (mri.get("side") or "").lower().strip()
    eeg_region = (eeg.get("region") or "").lower().strip()
    mri_lobe = (mri.get("lobar_location") or "").lower().strip()
    meaningful_sides = {"left","right"}
    if eeg_side in meaningful_sides and mri_side in meaningful_sides and eeg_side != mri_side:
        return "Discordant", (
            f"EEG lateralizes to {eeg_side} but MRI lesion is {mri_side}. "
            "Invasive EEG (SEEG) should be considered.")
    if has_eeg and has_mri:
        if eeg_region and mri_lobe and eeg_region not in mri_lobe and mri_lobe not in eeg_region:
            return "Partial concordance", f"EEG region ({eeg_region}) and MRI lobe ({mri_lobe}) do not clearly overlap."
        return "Strong concordance", "EEG and MRI are concordant in lateralization/region."
    if has_eeg and not has_mri:
        return "Partial (EEG only)", "MRI unavailable."
    return "Partial (MRI only)", "EEG unavailable."


def identify_missing_data(data: Dict) -> List[str]:
    missing = []
    if not data.get("eeg", {}).get("available"):
        missing.append("No EEG data — critical for classification")
    elif not data.get("eeg", {}).get("ictal_onset_described"):
        missing.append("No ictal EEG — video-EEG or SEEG would improve localization")
    if not data.get("mri", {}).get("available"):
        missing.append("No MRI data")
    elif data.get("mri", {}).get("result") == "Indeterminate":
        missing.append("MRI indeterminate — consider 3T epilepsy-protocol MRI")
    seizures = data.get("seizure_types", [])
    if not seizures:
        missing.append("No seizure semiology entered")
    else:
        for i, sz in enumerate(seizures):
            if not any([sz.get("seizure_description"), sz.get("aura_description"),
                        sz.get("motor_features"), sz.get("aura")]):
                missing.append(f"Seizure type {i+1}: no description")
    if not data.get("current_age"):
        missing.append("Patient age not specified")
    if not data.get("onset_age"):
        missing.append("Onset age not specified")
    return missing
