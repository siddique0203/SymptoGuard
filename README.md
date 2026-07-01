# SymptoGuard

SymptoGuard is an academic Streamlit web application for symptom-based 100-disease prediction.

## Updated deployed model

- Final model: Tuned Bernoulli Naive Bayes
- Split: 70% training, 15% validation, 15% final test
- Final test accuracy: 89.60%
- Final test weighted F1-score: 89.66%
- Final test macro F1-score: 89.62%
- Disease classes: 100
- Final symptom features: 227
- Processed records: 101,903

## Run locally

```bash
pip install -r requirements.txt
streamlit run symptoguard_xai_app.py
```

## Required structure

```text
SymptoGuard-main/
├── symptoguard_xai_app.py
├── requirements.txt
├── data/
│   └── 100_Disease.csv
├── models/
│   ├── best_nb_model.pkl
│   ├── label_encoder.pkl
│   └── feature_columns.pkl
└── assets/
```

> Medical safety notice: This project is for academic research and preliminary health awareness only. It is not a medical diagnosis system.
