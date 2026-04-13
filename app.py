"""
app.py
EpiloClassify — Epilepsy Classification and Localization Decision-Support Tool
Main Streamlit application entry point.

Run with:  streamlit run app.py
"""

import streamlit as st
import json
import os

from utils import (
    initialize_session_state,
    validate_patient_data,
    export_report_as_markdown,
    collect_seizure_type_data,
    sanitize_text,
)
from literature_processing import extract_text_from_uploaded_file
from report_generator import generate_report

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EpiloClassify",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main palette */
    :root {
        --primary: #1a3a5c;
        --accent: #2e7d9e;
        --bg-light: #f8fafc;
        --text-main: #1e2d3d;
        --warn: #c0392b;
    }

    .main-header {
        background: linear-gradient(135deg, #1a3a5c 0%, #2e7d9e 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .main-header p { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }

    .disclaimer-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-left: 5px solid #e67e22;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 1rem;
        font-size: 0.87rem;
    }

    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a3a5c;
        border-bottom: 2px solid #2e7d9e;
        padding-bottom: 0.3rem;
        margin: 1.2rem 0 0.7rem;
    }

    .report-box {
        background: #f8fafc;
        border: 1px solid #d0dce8;
        border-radius: 8px;
        padding: 1.5rem;
        max-height: 75vh;
        overflow-y: auto;
    }

    .confidence-high { color: #27ae60; font-weight: bold; }
    .confidence-moderate { color: #e67e22; font-weight: bold; }
    .confidence-low { color: #c0392b; font-weight: bold; }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1a3a5c, #2e7d9e);
        color: white;
        border: none;
        font-weight: 600;
        padding: 0.6rem 2rem;
        font-size: 1rem;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2e7d9e, #1a3a5c);
    }

    div[data-testid="stExpander"] > details {
        border: 1px solid #c8d6e0;
        border-radius: 6px;
        background: #fafcfd;
    }
</style>
""", unsafe_allow_html=True)

# ── Initialize session state ──────────────────────────────────────────────────

initialize_session_state(st)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>🧠 EpiloClassify</h1>
    <p>Epilepsy Classification and Localization — Clinical Decision-Support Tool</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-box">
    ⚠️ <strong>CLINICAL DISCLAIMER:</strong> EpiloClassify is a decision-support and educational tool only.
    It does not constitute a medical diagnosis and does not replace the clinical judgment of a qualified 
    epileptologist or neurologist. All outputs must be critically reviewed by a clinician before clinical use.
</div>
""", unsafe_allow_html=True)

# ── Sidebar: API Key + Literature Upload ──────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # API Key
    api_key_env = os.environ.get("GROQ_API_KEY", "")
    api_key_input = st.text_input(
        "Groq API Key",
        value=api_key_env or st.session_state.get("api_key", ""),
        type="password",
        help="Your Groq API key (free at console.groq.com). Can also be set via GROQ_API_KEY environment variable.",
    )
    if api_key_input:
        st.session_state["api_key"] = api_key_input

    st.markdown("---")

    # Literature upload
    st.markdown("### 📚 Literature Upload")
    st.caption(
        "Upload PDF or text files of epilepsy articles. "
        "These are used as the primary evidence base for the report, "
        "in addition to the curated 340-article reference set."
    )
    uploaded_files = st.file_uploader(
        "Upload articles (PDF / TXT)",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        key="literature_upload",
    )

    if uploaded_files:
        st.session_state["uploaded_docs"] = []
        with st.spinner("Extracting text from uploaded literature..."):
            for uf in uploaded_files:
                try:
                    text = extract_text_from_uploaded_file(uf)
                    st.session_state["uploaded_docs"].append({
                        "name": uf.name,
                        "text": text,
                    })
                except Exception as e:
                    st.warning(f"Could not extract text from {uf.name}: {e}")
        st.success(f"✅ {len(st.session_state['uploaded_docs'])} document(s) loaded.")
    else:
        # Keep existing docs if no new upload
        if not st.session_state.get("uploaded_docs"):
            st.info("No articles uploaded. The curated 340-article reference set will be used.")

    st.markdown("---")
    st.markdown("### 📋 About")
    st.caption(
        "EpiloClassify v1.0  \n"
        "Built on ILAE 2017 classification.  \n"
        "LLM: Groq — Llama 3.3 70b (free).  \n"
        "Reference dataset: 340 peer-reviewed epilepsy articles."
    )


# ── Main form ─────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📝 Patient Data Entry", "📊 Generate Report", "ℹ️ Help & Reference"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Patient Data Entry
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── A. Demographics ───────────────────────────────────────────────────────
    with col_left:
        st.markdown('<div class="section-header">A. Demographics & Background</div>', unsafe_allow_html=True)

        current_age = st.number_input("Current age (years)", min_value=0, max_value=120, value=None,
                                       placeholder="e.g. 35", key="current_age_input")
        sex = st.selectbox("Sex", ["Not specified", "Male", "Female", "Other"], key="sex_input")
        handedness = st.selectbox("Handedness", ["Not specified", "Right-handed", "Left-handed", "Ambidextrous"],
                                   key="hand_input")
        onset_age = st.number_input("Age at epilepsy onset (years)", min_value=0, max_value=120,
                                     value=None, placeholder="e.g. 18", key="onset_age_input")

    # ── B. Risk Factors ───────────────────────────────────────────────────────
    with col_right:
        st.markdown('<div class="section-header">B. Epilepsy Risk Factors</div>', unsafe_allow_html=True)
        rf_col1, rf_col2 = st.columns(2)
        with rf_col1:
            rf_perinatal = st.checkbox("Perinatal insult", key="rf_perinatal")
            rf_febrile = st.checkbox("Febrile seizures", key="rf_febrile")
            rf_cns = st.checkbox("CNS infection", key="rf_cns")
            rf_trauma = st.checkbox("Head trauma", key="rf_trauma")
            rf_stroke = st.checkbox("Stroke", key="rf_stroke")
        with rf_col2:
            rf_tumor = st.checkbox("Brain tumor", key="rf_tumor")
            rf_dev = st.checkbox("Developmental delay", key="rf_dev")
            rf_family = st.checkbox("Family history of epilepsy", key="rf_family")
            rf_genetic = st.checkbox("Known genetic syndrome", key="rf_genetic")
            rf_surgery = st.checkbox("Prior neurosurgery", key="rf_surgery")
        rf_other = st.text_input("Other risk factors (free text)", key="rf_other")

    # ── C. Seizure Burden ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">C. Seizure Burden</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sz_frequency = st.selectbox(
            "Seizure frequency",
            ["Not specified", "Multiple per day", "Daily", "Weekly", "Monthly",
             "Several per year", "Rare (<1/year)", "Seizure-free"],
            key="sz_freq"
        )
    with c2:
        time_last = st.text_input("Time since last seizure", placeholder="e.g. 2 weeks", key="time_last")
    with c3:
        disorder_dur = st.number_input("Duration of disorder (years)", min_value=0.0,
                                        max_value=80.0, value=None, step=0.5,
                                        placeholder="years", key="disorder_dur")
    with c4:
        drug_resistant = st.selectbox("Drug-resistant epilepsy suspected?",
                                       ["Unknown", "Yes", "No"], key="drug_resist")
    current_asms = st.text_input(
        "Current ASMs (anti-seizure medications, free text)",
        placeholder="e.g. Levetiracetam 1000mg BID, Lamotrigine 150mg BID",
        key="current_asms"
    )

    # ── D. Seizure Semiology ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">D. Seizure Semiology</div>', unsafe_allow_html=True)

    n_seizure_types = st.radio(
        "Number of seizure types",
        [1, 2, 3, 4],
        horizontal=True,
        key="n_sz_types"
    )

    seizure_data_list = []

    for sz_idx in range(n_seizure_types):
        with st.expander(f"🔶 Seizure Type {sz_idx + 1}", expanded=(sz_idx == 0)):
            sz_col1, sz_col2 = st.columns([1, 1], gap="medium")

            with sz_col1:
                sz_desc = st.text_area(
                    "Seizure description (free text)",
                    placeholder="Describe the seizure from beginning to end...",
                    height=100,
                    key=f"sz_desc_{sz_idx}"
                )
                has_aura = st.checkbox("Aura present", key=f"sz_aura_{sz_idx}")
                aura_desc = ""
                if has_aura:
                    aura_desc = st.text_input(
                        "Aura description",
                        placeholder="e.g. rising epigastric sensation, déjà vu, tingling in right hand",
                        key=f"sz_aura_desc_{sz_idx}"
                    )
                awareness = st.selectbox(
                    "Awareness during seizure",
                    ["Unknown", "Preserved", "Impaired"],
                    key=f"sz_aware_{sz_idx}"
                )
                duration = st.text_input(
                    "Typical seizure duration",
                    placeholder="e.g. 90 seconds",
                    key=f"sz_dur_{sz_idx}"
                )
                motor_feat = st.text_area(
                    "Motor features (free text)",
                    placeholder="e.g. right arm tonic posturing, bilateral clonic jerks...",
                    height=60,
                    key=f"sz_motor_{sz_idx}"
                )
                nonmotor_feat = st.text_area(
                    "Non-motor features (free text)",
                    placeholder="e.g. staring, unresponsiveness...",
                    height=60,
                    key=f"sz_nonmotor_{sz_idx}"
                )

            with sz_col2:
                st.markdown("**Specific features:**")
                feat_col1, feat_col2 = st.columns(2)
                with feat_col1:
                    automatisms = st.checkbox("Automatisms", key=f"sz_auto_{sz_idx}")
                    behavioral_arrest = st.checkbox("Behavioral arrest", key=f"sz_ba_{sz_idx}")
                    speech_arrest = st.checkbox("Speech arrest (ictal)", key=f"sz_sa_{sz_idx}")
                    head_version = st.checkbox("Head version", key=f"sz_hv_{sz_idx}")
                    eye_deviation = st.checkbox("Eye deviation", key=f"sz_ed_{sz_idx}")
                    unilateral_tonic = st.checkbox("Unilateral tonic posturing", key=f"sz_ut_{sz_idx}")
                with feat_col2:
                    clonic = st.checkbox("Clonic movements", key=f"sz_cl_{sz_idx}")
                    hypermotor = st.checkbox("Hypermotor features", key=f"sz_hm_{sz_idx}")
                    postictal_conf = st.checkbox("Postictal confusion", key=f"sz_pc_{sz_idx}")
                    postictal_aph = st.checkbox("Postictal aphasia", key=f"sz_pa_{sz_idx}")
                    todds = st.checkbox("Todd's paresis", key=f"sz_tp_{sz_idx}")
                    nocturnal = st.checkbox("Nocturnal predominance", key=f"sz_noc_{sz_idx}")
                    clustering = st.checkbox("Clustering", key=f"sz_cl2_{sz_idx}")

                # Symptom free text fields
                sensory_sym = st.text_input("Sensory symptoms", placeholder="e.g. tingling right hand",
                                             key=f"sz_sens_{sz_idx}")
                autonomic_sym = st.text_input("Autonomic symptoms", placeholder="e.g. tachycardia, sweating",
                                               key=f"sz_auto2_{sz_idx}")
                emotional_sym = st.text_input("Emotional symptoms", placeholder="e.g. fear, déjà vu, anxiety",
                                               key=f"sz_emo_{sz_idx}")
                cognitive_sym = st.text_input("Cognitive symptoms", placeholder="e.g. forced thinking, memory lapse",
                                               key=f"sz_cog_{sz_idx}")

                triggered = st.checkbox("Seizure triggered by specific stimulus", key=f"sz_trig_{sz_idx}")
                trigger_desc = ""
                if triggered:
                    trigger_desc = st.text_input("Trigger description", key=f"sz_trig_desc_{sz_idx}")

                expert_sz_summary = st.text_area(
                    "Clinician impression / expert summary (optional)",
                    placeholder="Any additional clinical context for this seizure type...",
                    height=60,
                    key=f"sz_expert_{sz_idx}"
                )

            seizure_data_list.append({
                "seizure_description": sz_desc,
                "aura": has_aura,
                "aura_description": aura_desc,
                "awareness": awareness,
                "duration": duration,
                "motor_features": motor_feat,
                "nonmotor_features": nonmotor_feat,
                "automatisms": automatisms,
                "behavioral_arrest": behavioral_arrest,
                "speech_arrest": speech_arrest,
                "head_version": head_version,
                "eye_deviation": eye_deviation,
                "unilateral_tonic": unilateral_tonic,
                "clonic_movements": clonic,
                "hypermotor_features": hypermotor,
                "postictal_confusion": postictal_conf,
                "postictal_aphasia": postictal_aph,
                "todds_paresis": todds,
                "nocturnal": nocturnal,
                "clustering": clustering,
                "sensory_symptoms": sensory_sym,
                "autonomic_symptoms": autonomic_sym,
                "emotional_symptoms": emotional_sym,
                "cognitive_symptoms": cognitive_sym,
                "triggered": triggered,
                "trigger_description": trigger_desc,
                "expert_summary": expert_sz_summary,
            })

    # ── E. EEG Data ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">E. EEG Data</div>', unsafe_allow_html=True)

    eeg_available = st.checkbox("EEG data available", value=True, key="eeg_avail")

    eeg_data = {"available": eeg_available}

    if eeg_available:
        eeg_col1, eeg_col2 = st.columns(2)
        with eeg_col1:
            eeg_types = st.multiselect(
                "EEG type(s) performed",
                ["Routine EEG", "Sleep EEG", "Prolonged EEG", "Video-EEG",
                 "Ambulatory EEG", "Invasive EEG (ECoG/SEEG)"],
                key="eeg_types"
            )
            eeg_result = st.selectbox("Overall EEG result", ["Not specified", "Normal", "Abnormal"], key="eeg_result")
            eeg_desc = st.text_area(
                "EEG description (free text)",
                placeholder="Describe the EEG findings in detail...",
                height=80,
                key="eeg_desc"
            )
        with eeg_col2:
            st.markdown("**Structured EEG findings:**")
            focal_epilep = st.checkbox("Focal epileptiform discharges", key="eeg_focal")
            gen_discharges = st.checkbox("Generalized discharges", key="eeg_gen")
            focal_slow = st.checkbox("Focal slowing", key="eeg_slow")
            lat_abnorm = st.checkbox("Lateralized abnormality", key="eeg_lat")
            if focal_epilep or focal_slow or lat_abnorm:
                eeg_side = st.selectbox("Side", ["Not specified", "Left", "Right", "Bilateral"], key="eeg_side")
                eeg_region = st.selectbox(
                    "Region",
                    ["Not specified", "Temporal", "Frontal", "Parietal", "Occipital",
                     "Central", "Temporal-frontal", "Temporal-parietal", "Multilobar"],
                    key="eeg_region"
                )
            else:
                eeg_side = "Not specified"
                eeg_region = "Not specified"

            ictal_onset = st.checkbox("Ictal onset described (video-EEG or icEEG)", key="eeg_ictal")
            ictal_desc = ""
            if ictal_onset:
                ictal_desc = st.text_area("Ictal EEG description", height=60,
                                           placeholder="Describe ictal onset pattern...",
                                           key="eeg_ictal_desc")

        eeg_data.update({
            "eeg_types": eeg_types,
            "result": eeg_result,
            "eeg_description": eeg_desc,
            "focal_epileptiform": focal_epilep,
            "generalized_discharges": gen_discharges,
            "focal_slowing": focal_slow,
            "lateralized_abnormality": lat_abnorm,
            "side": eeg_side,
            "region": eeg_region,
            "ictal_onset_described": ictal_onset,
            "ictal_eeg_description": ictal_desc,
        })

    # ── F. MRI Data ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">F. MRI Data</div>', unsafe_allow_html=True)

    mri_available = st.checkbox("MRI data available", value=True, key="mri_avail")
    mri_data = {"available": mri_available}

    if mri_available:
        mri_col1, mri_col2 = st.columns(2)
        with mri_col1:
            mri_result = st.selectbox("MRI result", ["Not specified", "Normal", "Abnormal", "Indeterminate"],
                                       key="mri_result")
            mri_desc = st.text_area(
                "MRI description (free text)",
                placeholder="Describe MRI findings in detail (sequence, location, extent)...",
                height=80,
                key="mri_desc"
            )
        with mri_col2:
            st.markdown("**Lesion category:**")
            lesion_options = [
                "Mesial temporal sclerosis", "Focal cortical dysplasia",
                "Tumor", "Vascular lesion", "Post-traumatic lesion",
                "Encephalomalacia", "Malformation of cortical development",
                "Hippocampal abnormality", "Diffuse abnormality", "Other"
            ]
            lesion_cats = st.multiselect("Select applicable categories", lesion_options, key="mri_lesion")
            mri_side = st.selectbox("Side", ["Not specified", "Left", "Right", "Bilateral"], key="mri_side")
            mri_lobe = st.selectbox(
                "Lobar location",
                ["Not specified", "Temporal", "Frontal", "Parietal", "Occipital",
                 "Insula", "Cingulate", "Basal ganglia", "Multilobar", "Diffuse"],
                key="mri_lobe"
            )
            mri_multifocal = st.checkbox("Multifocal lesion(s)", key="mri_multi")

        mri_data.update({
            "result": mri_result,
            "mri_description": mri_desc,
            "lesion_categories": lesion_cats,
            "side": mri_side,
            "lobar_location": mri_lobe,
            "multifocal": mri_multifocal,
        })

    # ── G. Expert Override ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">G. Clinician Impression / Expert Override (Optional)</div>',
                unsafe_allow_html=True)
    expert_override = st.text_area(
        "Expert clinical impression",
        placeholder="Any additional clinical context, pre-existing diagnosis, surgical planning considerations...",
        height=80,
        key="expert_override"
    )

    # ── Assemble patient data dict ────────────────────────────────────────────
    patient_data = {
        "current_age": current_age,
        "sex": sex,
        "handedness": handedness,
        "onset_age": onset_age,
        "risk_factors": {
            "perinatal_insult": rf_perinatal,
            "febrile_seizures": rf_febrile,
            "cns_infection": rf_cns,
            "head_trauma": rf_trauma,
            "stroke": rf_stroke,
            "brain_tumor": rf_tumor,
            "developmental_delay": rf_dev,
            "family_history": rf_family,
            "known_genetic_syndrome": rf_genetic,
            "prior_neurosurgery": rf_surgery,
            "other_text": rf_other,
        },
        "seizure_frequency": sz_frequency,
        "time_since_last_seizure": time_last,
        "disorder_duration": disorder_dur,
        "drug_resistant": drug_resistant,
        "current_asms": current_asms,
        "seizure_types": [collect_seizure_type_data(sz) for sz in seizure_data_list],
        "eeg": eeg_data,
        "mri": mri_data,
        "expert_override": expert_override,
    }
    st.session_state["patient_data"] = patient_data


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Generate Report
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### 🔬 Generate Classification and Localization Report")

    # Validation summary
    data = st.session_state.get("patient_data", {})
    warnings = validate_patient_data(data)
    if warnings:
        st.warning("**Data entry notes:**\n" + "\n".join(f"- {w}" for w in warnings))

    # Quick summary of entered data
    with st.expander("📋 Patient data summary (preview)", expanded=False):
        if data.get("current_age") or data.get("onset_age"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Age", data.get("current_age", "—"))
                st.metric("Sex", data.get("sex", "—"))
            with col_b:
                st.metric("Onset age", data.get("onset_age", "—"))
                st.metric("Handedness", data.get("handedness", "—"))
            with col_c:
                sz_count = len([s for s in data.get("seizure_types", []) if s.get("seizure_description") or s.get("aura")])
                st.metric("Seizure types described", sz_count)
                st.metric("Drug-resistant", data.get("drug_resistant", "—"))
        else:
            st.info("Complete the Patient Data Entry tab first.")

    # API key check
    api_key = st.session_state.get("api_key", "")
    if not api_key:
        st.error("⚠️ Please enter your Groq API key in the sidebar. Get one free at console.groq.com")
        st.stop()

    # Generate button
    generate_btn = st.button(
        "🧠 Generate Classification and Localization Report",
        type="primary",
        use_container_width=True,
    )

    if generate_btn:
        data = st.session_state.get("patient_data", {})
        uploaded_docs = st.session_state.get("uploaded_docs", [])

        with st.spinner("🔄 Analyzing patient data and generating expert report... (this may take 30–60 seconds)"):
            try:
                report = generate_report(
                    data=data,
                    uploaded_docs=uploaded_docs,
                    api_key=api_key,
                )
                st.session_state["generated_report"] = report
                st.session_state["report_error"] = None
            except Exception as e:
                st.session_state["report_error"] = str(e)
                st.session_state["generated_report"] = None

    # Display report or error
    if st.session_state.get("report_error"):
        st.error(f"❌ Error generating report: {st.session_state['report_error']}")
        st.info("Please check your API key and try again.")

    elif st.session_state.get("generated_report"):
        report_md = st.session_state["generated_report"]

        st.success("✅ Report generated successfully.")

        # Display tabs for the report
        report_tab1, report_tab2 = st.tabs(["📄 Report", "📥 Export"])

        with report_tab1:
            st.markdown('<div class="report-box">', unsafe_allow_html=True)
            st.markdown(report_md)
            st.markdown('</div>', unsafe_allow_html=True)

        with report_tab2:
            full_export = export_report_as_markdown(
                report_md,
                st.session_state.get("patient_data", {})
            )
            st.download_button(
                label="⬇️ Download as Markdown (.md)",
                data=full_export,
                file_name="epilepsy_classification_report.md",
                mime="text/markdown",
            )
            st.download_button(
                label="⬇️ Download as Text (.txt)",
                data=full_export,
                file_name="epilepsy_classification_report.txt",
                mime="text/plain",
            )
            st.markdown("#### Report preview (plain text):")
            st.text_area(
                "Report content",
                value=full_export,
                height=400,
                label_visibility="collapsed",
            )

    else:
        st.info(
            "Complete the **Patient Data Entry** tab, then click **Generate Report** above.\n\n"
            "The report will be generated using the Anthropic API with clinical reasoning grounded "
            "in the curated epilepsy literature reference set."
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Help & Reference
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### ℹ️ Help & Reference")

    with st.expander("🎯 How to use EpiloClassify", expanded=True):
        st.markdown("""
**Step-by-step guide:**

1. **Enter your API key** in the sidebar (Anthropic API key required).
2. **Upload literature** (optional) — PDF or TXT articles to augment the built-in 340-article reference set.
3. **Complete patient data** in the Patient Data Entry tab:
   - Demographics and epilepsy background
   - Risk factors (checkboxes)
   - Seizure burden (frequency, drug resistance)
   - Seizure semiology (1–4 seizure types with detailed features)
   - EEG data (structured + free text)
   - MRI data (structured + free text)
4. **Generate the report** from the Generate Report tab.
5. **Review and export** the structured report as markdown or text.

**Important notes:**
- The more detailed the seizure semiology and imaging/EEG data, the more specific the output.
- Vague or minimal data will produce conservative, lower-confidence classifications.
- Always review all outputs critically before any clinical use.
        """)

    with st.expander("📚 Classification framework (ILAE 2017)"):
        st.markdown("""
**Epilepsy types:**
- Focal epilepsy
- Generalized epilepsy
- Combined generalized and focal epilepsy
- Unknown epilepsy

**Seizure types (selected):**
- Focal aware seizure (FAS)
- Focal impaired awareness seizure (FIAS)
- Focal to bilateral tonic-clonic seizure (FBTCS)
- Generalized tonic-clonic seizure (GTCS)
- Absence seizure
- Tonic seizure
- Hypermotor seizure

**Localization:**
- Mesial temporal lobe
- Lateral temporal lobe
- Frontal lobe (including SMA, cingulate, opercular)
- Parietal lobe
- Occipital lobe
- Insular / insulo-opercular
- Multifocal / bilateral network
- Undetermined
        """)

    with st.expander("📖 Curated reference set (sample)"):
        from literature_processing import CURATED_REFERENCES
        import pandas as pd
        df = pd.DataFrame(CURATED_REFERENCES)
        st.dataframe(
            df[["pmid", "title", "authors", "journal", "year"]],
            use_container_width=True,
            height=400,
        )

    with st.expander("⚠️ Limitations and disclaimers"):
        st.markdown("""
**Limitations:**
- This tool does not perform autonomous diagnosis.
- Classification accuracy depends heavily on data completeness and quality.
- The LLM may produce plausible-sounding but incorrect reasoning — all outputs require clinician review.
- The tool does not have access to actual imaging or EEG recordings.
- Literature citations are drawn from a curated reference set; not all relevant literature may be represented.
- The tool is not validated for clinical use and should not be used as the basis for treatment decisions.

**When to be especially cautious:**
- Discordant EEG and MRI findings
- Vague or non-specific semiology
- Multiple possible etiologies
- Pediatric patients (age-dependent semiology is complex)
- Genetic syndromes with complex phenotypes
- Post-surgical or multifocal epilepsy
        """)
