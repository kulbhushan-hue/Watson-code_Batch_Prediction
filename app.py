"""
Streamlit dashboard for the SMF Batch Job Failure-Risk model.

Run locally:
    streamlit run app.py

Expects `job_failure_model.pkl` (built by train_model.py) in the
same folder as this file.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------
st.set_page_config(
    page_title="SMF Batch Job Failure Risk",
    page_icon="🖥️",
    layout="wide",
)


@st.cache_resource
def load_artifact(path: str = "job_failure_model.pkl"):
    return joblib.load(path)


try:
    artifact = load_artifact()
except FileNotFoundError:
    st.error(
        "Couldn't find `job_failure_model.pkl` in the app folder. "
        "Run `python train_model.py` first to create it."
    )
    st.stop()

model = artifact["model"]
le_servclass = artifact["label_encoder_servclass"]
FEATURE_COLS = artifact["feature_cols"]
NUMERIC_FEATURES = artifact["numeric_features"]
job_profile = artifact["job_profile"]
global_defaults = artifact["global_defaults"]
serv_class_options = artifact["serv_class_options"]
class_options = artifact["class_options"]
job_name_options = artifact["job_name_options"]
metrics = artifact["training_metrics"]
raw_df = artifact["raw_df"]


def encode_serv_class(value: str) -> int:
    """LabelEncoder-safe transform; falls back to 0 for unseen classes."""
    if value in le_servclass.classes_:
        return int(le_servclass.transform([value])[0])
    return 0


def make_feature_row(values: dict) -> pd.DataFrame:
    row = {c: values.get(c, 0) for c in FEATURE_COLS}
    return pd.DataFrame([row], columns=FEATURE_COLS)


def predict_risk(values: dict):
    X = make_feature_row(values)
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(model.predict(X)[0])
    return pred, proba


# --------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------
st.sidebar.title("🖥️ SMF Job Monitor")
page = st.sidebar.radio(
    "Go to",
    ["Predict a Job", "Batch Prediction (CSV)", "Dashboard", "Model Info"],
)

st.sidebar.markdown("---")
st.sidebar.metric("Model ROC AUC", f"{metrics['roc_auc']:.3f}" if metrics["roc_auc"] else "n/a")
st.sidebar.metric("Historical failure-risk rate", f"{metrics['label_positive_rate']*100:.1f}%")
st.sidebar.caption(f"Trained on {metrics['n_train']} jobs · tested on {metrics['n_test']} jobs")

# --------------------------------------------------------------------
# PAGE 1 — Predict a single job
# --------------------------------------------------------------------
if page == "Predict a Job":
    st.title("Predict Failure Risk for a Batch Job")
    st.caption(
        "Enter job metrics manually, or pick a known JOB_NAME to pre-fill "
        "its historical averages, then adjust as needed."
    )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        job_choice = st.selectbox(
            "Pre-fill from a known job (optional)",
            ["— none, start blank —"] + job_name_options,
        )

    if job_choice != "— none, start blank —":
        profile_row = job_profile.loc[job_profile["JOB_NAME"] == job_choice].iloc[0]
        defaults = profile_row.to_dict()
        default_serv_class = defaults.get("SERV_CLASS", serv_class_options[0])
        st.info(
            f"**{job_choice}** has run {int(profile_row['RUN_COUNT'])} times historically, "
            f"with an average failure-risk rate of {profile_row['FAILURE_RISK']*100:.0f}%."
        )
    else:
        defaults = global_defaults
        default_serv_class = serv_class_options[0]

    st.subheader("Job parameters")
    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            serv_class = st.selectbox(
                "Service class (SERV_CLASS)",
                serv_class_options,
                index=serv_class_options.index(default_serv_class)
                if default_serv_class in serv_class_options else 0,
            )
            job_class = st.selectbox(
                "CLASS",
                class_options,
                index=class_options.index(defaults.get("CLASS", class_options[0]))
                if defaults.get("CLASS", class_options[0]) in class_options else 0,
            )
            start_hour = st.slider("Start hour of day", 0, 23, int(defaults.get("START_HOUR", 12)) if "START_HOUR" in defaults else 12)
            start_dow = st.slider("Start day of week (0=Mon)", 0, 6, int(defaults.get("START_DOW", 2)) if "START_DOW" in defaults else 2)

        with c2:
            excp = st.number_input("EXCP", min_value=0.0, value=float(defaults.get("EXCP", global_defaults["EXCP"])))
            io_conn_sec = st.number_input("IO_CONN_SEC", min_value=0.0, value=float(defaults.get("IO_CONN_SEC", global_defaults["IO_CONN_SEC"])))
            ssch = st.number_input("SSCH", min_value=0.0, value=float(defaults.get("SSCH", global_defaults["SSCH"])))
            serv_unit = st.number_input("SERV_UNIT", min_value=0.0, value=float(defaults.get("SERV_UNIT", global_defaults["SERV_UNIT"])))
            mso_unit = st.number_input("MSO_UNIT", min_value=0.0, value=float(defaults.get("MSO_UNIT", global_defaults["MSO_UNIT"])))

        with c3:
            tcb_cpu_sec = st.number_input("TCB_CPU_SEC", min_value=0.0, value=float(defaults.get("TCB_CPU_SEC", global_defaults["TCB_CPU_SEC"])))
            srb_cpu_sec = st.number_input("SRB_CPU_SEC", min_value=0.0, value=float(defaults.get("SRB_CPU_SEC", global_defaults["SRB_CPU_SEC"])))
            cpu_sec = st.number_input("CPU_SEC", min_value=0.0, value=float(defaults.get("CPU_SEC", global_defaults["CPU_SEC"])))
            page_in = st.number_input("PAGE_IN", min_value=0.0, value=float(defaults.get("PAGE_IN", global_defaults["PAGE_IN"])))
            page_out = st.number_input("PAGE_OUT", min_value=0.0, value=float(defaults.get("PAGE_OUT", global_defaults["PAGE_OUT"])))
            page_swap = st.number_input("PAGE_SWAP", min_value=0.0, value=float(defaults.get("PAGE_SWAP", global_defaults["PAGE_SWAP"])))
            total_queue_sec = st.number_input("TOTAL_QUEUE_SEC (JQ_SEC + HQ_SEC)", min_value=0.0, value=float(defaults.get("TOTAL_QUEUE_SEC", global_defaults["TOTAL_QUEUE_SEC"])))

        submitted = st.form_submit_button("Predict risk", type="primary", use_container_width=True)

    if submitted:
        values = {
            "EXCP": excp, "IO_CONN_SEC": io_conn_sec, "PAGE_IN": page_in,
            "PAGE_OUT": page_out, "PAGE_SWAP": page_swap, "SSCH": ssch,
            "TCB_CPU_SEC": tcb_cpu_sec, "SRB_CPU_SEC": srb_cpu_sec,
            "SERV_UNIT": serv_unit, "MSO_UNIT": mso_unit,
            "TOTAL_QUEUE_SEC": total_queue_sec, "CPU_SEC": cpu_sec,
            "START_HOUR": start_hour, "START_DOW": start_dow, "CLASS": job_class,
            "SERV_CLASS_ENC": encode_serv_class(serv_class),
        }
        pred, proba = predict_risk(values)

        st.markdown("---")
        r1, r2 = st.columns([1, 2])
        with r1:
            if pred == 1:
                st.error(f"⚠️ HIGH RISK — {proba*100:.1f}% predicted failure-risk probability")
            else:
                st.success(f"✅ LOW RISK — {proba*100:.1f}% predicted failure-risk probability")
            st.progress(min(max(proba, 0.0), 1.0))
        with r2:
            st.write("**Top drivers for this model overall:**")
            importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False).head(5)
            st.bar_chart(importances)

# --------------------------------------------------------------------
# PAGE 2 — Batch prediction from an uploaded CSV
# --------------------------------------------------------------------
elif page == "Batch Prediction (CSV)":
    st.title("Batch Prediction from CSV")
    st.caption(
        "Upload a CSV of jobs (same raw columns as df_smf.csv — including "
        "SERV_CLASS, START_DTSTR, JQ_SEC, HQ_SEC, etc.) to score them all at once."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)

            # Recreate the same engineered features used at training time
            if "START_DTSTR" in new_df.columns:
                new_df["START_DTSTR"] = pd.to_datetime(
                    new_df["START_DTSTR"], format="%Y-%m-%d-%H.%M.%S.%f", errors="coerce"
                )
                new_df["START_HOUR"] = new_df["START_DTSTR"].dt.hour
                new_df["START_DOW"] = new_df["START_DTSTR"].dt.dayofweek

            if {"JQ_SEC", "HQ_SEC"}.issubset(new_df.columns):
                new_df["TOTAL_QUEUE_SEC"] = new_df["JQ_SEC"] + new_df["HQ_SEC"]

            if "SERV_CLASS" in new_df.columns:
                new_df["SERV_CLASS_ENC"] = new_df["SERV_CLASS"].apply(encode_serv_class)

            missing = [c for c in FEATURE_COLS if c not in new_df.columns]
            for c in missing:
                new_df[c] = global_defaults.get(c, 0)
            if missing:
                st.warning(f"Missing columns filled with training-set medians: {missing}")

            X_new = new_df[FEATURE_COLS].fillna(0)
            new_df["FAILURE_RISK_PRED"] = model.predict(X_new)
            new_df["FAILURE_RISK_PROBA"] = model.predict_proba(X_new)[:, 1]

            st.success(f"Scored {len(new_df)} jobs.")
            n_high = int((new_df["FAILURE_RISK_PRED"] == 1).sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Jobs scored", len(new_df))
            c2.metric("Flagged high-risk", n_high)
            c3.metric("High-risk rate", f"{n_high/len(new_df)*100:.1f}%")

            show_cols = [c for c in ["JOB_NAME", "SERV_CLASS", "CLASS"] if c in new_df.columns]
            show_cols += ["FAILURE_RISK_PRED", "FAILURE_RISK_PROBA"]
            st.dataframe(
                new_df[show_cols].sort_values("FAILURE_RISK_PROBA", ascending=False),
                use_container_width=True,
            )

            st.download_button(
                "Download scored CSV",
                new_df.to_csv(index=False).encode("utf-8"),
                file_name="scored_jobs.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Couldn't process this file: {e}")
    else:
        st.info("Waiting for a CSV upload.")

# --------------------------------------------------------------------
# PAGE 3 — Dashboard / analytics over the training data
# --------------------------------------------------------------------
elif page == "Dashboard":
    st.title("Batch Job Analytics Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total jobs (history)", len(raw_df))
    c2.metric("Unique job names", raw_df["JOB_NAME"].nunique())
    c3.metric("Service classes", raw_df["SERV_CLASS"].nunique())
    c4.metric("High-risk jobs", int((raw_df["FAILURE_RISK"] == 1).sum()))

    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("Failure risk by service class")
        risk_by_class = raw_df.groupby("SERV_CLASS")["FAILURE_RISK"].mean().sort_values(ascending=False)
        st.bar_chart(risk_by_class)

    with right:
        st.subheader("Elapsed time distribution")
        st.bar_chart(
            raw_df["ELAPSED_SEC"].clip(upper=raw_df["ELAPSED_SEC"].quantile(0.95))
            .value_counts(bins=15).sort_index()
        )

    st.markdown("---")
    left2, right2 = st.columns(2)
    with left2:
        st.subheader("Jobs by start hour")
        st.bar_chart(raw_df["START_HOUR"].value_counts().sort_index())
    with right2:
        st.subheader("Average queue time by service class")
        st.bar_chart(raw_df.groupby("SERV_CLASS")["TOTAL_QUEUE_SEC"].mean().sort_values(ascending=False))

    st.markdown("---")
    st.subheader("Highest-risk jobs on record")
    top_risky = job_profile.sort_values("FAILURE_RISK", ascending=False).head(10)
    st.dataframe(
        top_risky[["JOB_NAME", "SERV_CLASS", "RUN_COUNT", "FAILURE_RISK", "ELAPSED_SEC"]],
        use_container_width=True,
    )

# --------------------------------------------------------------------
# PAGE 4 — Model info
# --------------------------------------------------------------------
elif page == "Model Info":
    st.title("Model Information")

    c1, c2, c3 = st.columns(3)
    c1.metric("ROC AUC", f"{metrics['roc_auc']:.3f}" if metrics["roc_auc"] else "n/a")
    c2.metric("Training rows", metrics["n_train"])
    c3.metric("Test rows", metrics["n_test"])

    st.subheader("Feature importance")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    st.bar_chart(importances)
    st.dataframe(importances.rename("importance").reset_index().rename(columns={"index": "feature"}))

    st.subheader("Model configuration")
    st.json({k: v for k, v in model.get_params().items()})

    st.caption(
        "FAILURE_RISK is a proxy label: the top 20% of jobs by a weighted "
        "rank of elapsed time, queue time, and EXCP/CPU ratio (computed "
        "within each service class) are labelled high-risk."
    )
