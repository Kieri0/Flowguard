# FlowGuard Streamlit app

This folder is a complete Streamlit deployment for the corrected 20-feature
`model.pkl`. It contains `app.py`, the model, two example flows, and pinned
Python dependencies. The training CSV and notebooks are not needed online.

## Run locally

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -m streamlit run app.py
```

## Deploy on Streamlit Community Cloud

Upload the **contents** of this folder to a GitHub repository, then create a
Streamlit app with `app.py` as the main file. Keep `model.pkl`,
`requirements.txt`, and the `examples` folder beside `app.py`.

The manual form gets its exact 20 field names from `model.pkl`. The CSV uploader
accepts those fields in any order and ignores extra columns. It classifies all
rows and lets you download a results CSV.

The bundled examples are the first complete BENIGN and DDoS rows from the
study's held-out test split. They are illustrations, not new validation data.
The models were trained on one CICIDS2017 study capture and are not live
network monitoring or a standalone security control.
