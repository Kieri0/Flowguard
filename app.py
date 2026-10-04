"""FlowGuard Streamlit demo for the study's saved 20-feature models."""

import csv
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
MODEL_NAMES = {
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
}

CSS = """
<style>
  .stApp { background: #fffaf7; color: #262238; }
  .block-container { max-width: 1180px; padding-top: 2rem; }
  .hero {
    position: relative; overflow: hidden; padding: 2.1rem 2.4rem;
    border-radius: 22px; background: #eee7fb; color: #2f2549;
    margin-bottom: 1.5rem;
  }
  .hero::after {
    content: "✿"; position: absolute; right: 1.8rem; top: -.8rem;
    font-size: 9rem; line-height: 1; color: #d8c8f1; transform: rotate(14deg);
  }
  .hero h1 { position: relative; z-index: 1; margin: 0; font-size: clamp(2.2rem, 5vw, 3.6rem); letter-spacing: -.03em; }
  .hero p { position: relative; z-index: 1; margin: .55rem 0 0; max-width: 55ch; font-size: 1.05rem; }
  .hero-meta { position: relative; z-index: 1; margin-top: 1.25rem; font-weight: 700; color: #554477; }
  div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {
    background: #714fb4; border-color: #714fb4; color: white;
  }
  div.stButton > button[kind="primary"]:hover, div.stFormSubmitButton > button[kind="primary"]:hover {
    background: #5a3b98; border-color: #5a3b98;
  }
  div.stButton > button:focus-visible, input:focus-visible { outline: 3px solid #ac81e8; outline-offset: 2px; }
  .result {
    padding: 1.2rem 1.35rem; border-radius: 16px; min-height: 145px;
    background: white; border: 1px solid #e3dced;
  }
  .result .model { color: #564b6e; font-weight: 700; margin-bottom: .65rem; }
  .result .label { font-size: 2rem; font-weight: 800; letter-spacing: -.025em; }
  .result.benign .label { color: #18745d; }
  .result.ddos .label { color: #b84457; }
  .result .detail { color: #655c73; margin-top: .55rem; font-size: .91rem; }
  .small-note { color: #5c5469; font-size: .92rem; }
</style>
"""


@st.cache_resource(show_spinner="Loading the study models…")
def load_bundle(modified_time: float) -> dict:
    # model.pkl is shipped with this app; never load a pickle uploaded by a visitor.
    with MODEL_PATH.open("rb") as file:
        bundle = pickle.load(file)
    features = bundle.get("features")
    models = bundle.get("models")
    if bundle.get("artifact_version") != 1 or not isinstance(features, list) or len(features) != 20:
        raise ValueError("model.pkl must contain the corrected 20-feature bundle.")
    if not isinstance(models, dict) or set(models) != set(MODEL_NAMES):
        raise ValueError("model.pkl must contain both study models.")
    for model in models.values():
        fitted_names = list(model.named_steps["imputer"].feature_names_in_)
        if fitted_names != features:
            raise ValueError("The model and its saved feature list do not match.")
    return bundle


def sample_values(kind: str, features: list[str]) -> dict[str, str]:
    path = ROOT / "examples" / f"{kind}.csv"
    with path.open(newline="", encoding="utf-8-sig") as file:
        row = next(csv.DictReader(file), None)
    if row is None or any(name not in row for name in features):
        raise ValueError(f"The {kind} example is missing required model fields.")
    return {name: row[name] for name in features}


def manual_frame(features: list[str]) -> pd.DataFrame:
    numbers = {}
    for name in features:
        raw = st.session_state.get(f"field_{name}", "").strip()
        try:
            number = float(raw)
        except ValueError as error:
            raise ValueError(f"Enter a number for {name}.") from error
        if not math.isfinite(number):
            raise ValueError(f"Enter a finite number for {name}.")
        numbers[name] = number
    return pd.DataFrame([numbers], columns=features)


def uploaded_frame(file, features: list[str]) -> pd.DataFrame:
    try:
        file.seek(0)
        header = pd.read_csv(file, nrows=0)
        names = [str(name).strip() for name in header.columns]
        if len(names) != len(set(names)):
            raise ValueError("The CSV contains repeated column names.")
        missing = [name for name in features if name not in names]
        if missing:
            raise ValueError(f"The CSV is missing {len(missing)} model fields. First missing: {missing[0]}.")
        file.seek(0)
        frame = pd.read_csv(file, usecols=lambda name: name.strip() in set(features))
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as error:
        raise ValueError("The CSV could not be read. Check its header and rows.") from error
    if frame.empty:
        raise ValueError("The CSV needs at least one data row.")
    frame.columns = frame.columns.str.strip()
    frame = frame[features].apply(pd.to_numeric, errors="coerce")
    invalid = np.argwhere(~np.isfinite(frame.to_numpy(dtype=float)))
    if len(invalid):
        row, column = invalid[0]
        raise ValueError(f"Data row {row + 1} has a missing or nonnumeric value for {features[column]}.")
    return frame


def predict(frame: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for key, model in bundle["models"].items():
        result[MODEL_NAMES[key]] = np.where(model.predict(frame) == 1, "DDoS", "BENIGN")
    return result.reset_index(drop=True)


def feature_group(name: str) -> str:
    if "IAT" in name:
        return "Timing"
    if "Header" in name or "Init_Win" in name or "Packets/s" in name:
        return "Headers and rate"
    if "Packet Length" in name or "Segment Size" in name:
        return "Packet sizes"
    return "Connection and volume"


def show_single_result(result: dict, bundle: dict) -> None:
    st.subheader("Model predictions")
    columns = st.columns(2, gap="medium")
    for column, key in zip(columns, MODEL_NAMES):
        label = result[MODEL_NAMES[key]]
        accuracy = bundle.get("evaluation", {}).get(key, {}).get("accuracy")
        detail = f"Study test accuracy: {accuracy:.2%}" if accuracy is not None else "Study model"
        with column:
            st.markdown(
                f'<div class="result {label.lower()}">'
                f'<div class="model">{MODEL_NAMES[key]}</div>'
                f'<div class="label">{label}</div>'
                f'<div class="detail">{detail}</div></div>',
                unsafe_allow_html=True,
            )
    if result["Random Forest"] != result["Logistic Regression"]:
        st.warning("The models disagree on this flow. Review the measurements before drawing a conclusion.")
    st.caption("One flow is classified as BENIGN or DDoS. This is a study demo, not live attack monitoring.")


def show_batch_result(result: pd.DataFrame, source: str) -> None:
    st.subheader("CSV results")
    st.caption(f"{source} · {len(result):,} flows")
    columns = st.columns(2)
    for column, model_name in zip(columns, MODEL_NAMES.values()):
        with column:
            st.metric(f"{model_name}: DDoS", f"{result[model_name].eq('DDoS').sum():,}")
    disagreements = result["Random Forest"].ne(result["Logistic Regression"]).sum()
    st.write(f"Models disagree on **{disagreements:,}** flows.")
    preview = result.copy()
    preview.insert(0, "CSV data row", np.arange(1, len(preview) + 1))
    st.dataframe(preview.head(100), hide_index=True, use_container_width=True)
    st.download_button(
        "Download all predictions",
        preview.to_csv(index=False).encode("utf-8"),
        file_name="flowguard_predictions.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="FlowGuard · DDoS flow check", page_icon="🌼", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        '<div class="hero"><h1>FlowGuard</h1>'
        '<p>Explore one network flow or upload a CSV, then compare two saved models side by side.</p>'
        '<div class="hero-meta">20 measurements · 2 models · one clear result</div></div>',
        unsafe_allow_html=True,
    )

    try:
        bundle = load_bundle(MODEL_PATH.stat().st_mtime)
    except (OSError, ValueError, pickle.UnpicklingError, ImportError, AttributeError) as error:
        st.error(f"Could not load model.pkl: {error}")
        st.stop()
    features = bundle["features"]

    st.subheader("Try an example")
    st.caption("Each button fills the 20 manual fields and immediately compares both models.")
    benign_button, ddos_button, _ = st.columns([1, 1, 2])
    chosen = None
    if benign_button.button("Load BENIGN example", use_container_width=True):
        chosen = "BENIGN"
    if ddos_button.button("Load DDoS example", use_container_width=True):
        chosen = "DDoS"
    if chosen:
        try:
            values = sample_values(chosen, features)
            for name, value in values.items():
                st.session_state[f"field_{name}"] = value
            st.session_state["entry_mode"] = "Manual inputs"
            st.session_state["loaded_example"] = chosen
            st.session_state["result"] = {
                "mode": "Manual inputs",
                "data": predict(pd.DataFrame([{name: float(values[name]) for name in features}], columns=features), bundle).iloc[0].to_dict(),
            }
        except (OSError, ValueError) as error:
            st.error(str(error))

    mode = st.radio("Choose input method", ["Manual inputs", "CSV upload"], horizontal=True, key="entry_mode")
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        if mode == "Manual inputs":
            st.subheader("Flow measurements")
            st.caption("All 20 fields use CICIDS2017 flow column names. Every field needs a numeric value.")
            if st.session_state.get("loaded_example"):
                st.success(f"{st.session_state['loaded_example']} example loaded into the text boxes below.")
            groups = {name: [] for name in ("Connection and volume", "Packet sizes", "Timing", "Headers and rate")}
            for name in features:
                groups[feature_group(name)].append(name)
            with st.form("manual_form"):
                for title, names in groups.items():
                    if not names:
                        continue
                    st.markdown(f"#### {title}")
                    columns = st.columns(2)
                    for index, name in enumerate(names):
                        columns[index % 2].text_input(name, key=f"field_{name}", placeholder="Enter a number")
                submitted = st.form_submit_button("Analyze this flow", type="primary")
            if submitted:
                try:
                    st.session_state["result"] = {
                        "mode": "Manual inputs",
                        "data": predict(manual_frame(features), bundle).iloc[0].to_dict(),
                    }
                    st.session_state.pop("loaded_example", None)
                except ValueError as error:
                    st.error(str(error))
        else:
            st.subheader("Upload flow data")
            st.caption("The CSV needs the 20 named model columns. Extra columns are ignored; all rows are classified.")
            uploaded = st.file_uploader("Choose a CSV file", type="csv", on_change=lambda: st.session_state.pop("result", None))
            if st.button("Analyze CSV", type="primary"):
                if uploaded is None:
                    st.error("Choose a CSV file first.")
                else:
                    try:
                        frame = uploaded_frame(uploaded, features)
                        st.session_state["result"] = {
                            "mode": "CSV upload",
                            "data": predict(frame, bundle),
                            "source": uploaded.name,
                        }
                    except ValueError as error:
                        st.error(str(error))
    with right:
        result = st.session_state.get("result")
        if result and result["mode"] == mode:
            if mode == "Manual inputs":
                show_single_result(result["data"], bundle)
            else:
                show_batch_result(result["data"], result["source"])
        else:
            st.subheader("Model predictions")
            st.info("Load an example, enter 20 measurements, or upload a CSV to see the two predictions here.")

    st.divider()
    st.markdown(
        '<p class="small-note">FlowGuard uses models trained on a single CICIDS2017 DDoS/BENIGN study capture. '
        'Results may differ on other networks and should not be used as a standalone security decision.</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
