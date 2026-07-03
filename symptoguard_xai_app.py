"""
SymptoGuard - Modern Responsive AI Symptom Checker

Run:
    streamlit run symptoguard_xai_app.py

Required files:
    models/best_nb_model.pkl
    models/label_encoder.pkl
    models/feature_columns.pkl
    data/100_Disease.csv

Optional team images:
    assets/abu_bakar.jpg
    assets/ishan.jpg
    assets/saif.jpg
    assets/fysal.jpg
"""

from pathlib import Path
import base64
import html
import mimetypes
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# FILE PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"

MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

MODEL_PATH = MODELS_DIR / "best_nb_model.pkl"
LE_PATH = MODELS_DIR / "label_encoder.pkl"
FEATURES_PATH = MODELS_DIR / "feature_columns.pkl"
DATA_PATH = DATA_DIR / "100_Disease.csv"

# ─────────────────────────────────────────────
# TEAM INFO
# Replace placeholder emails with real emails.
# Put images inside: F:\SymptoGuard\assets
# ─────────────────────────────────────────────
TEAM_MEMBERS = [
    {
        "name": "Md Abu Bakar Siddique",
        "email": "siddique0203@gmail.com",
        "image": "abu_bakar.jpg",
        "role": "Researcher & Developer",
    },
    {
        "name": "Ibnul Ishtiak Ishan",
        "email": "ishanibnul102913@gmail.com",
        "image": "ishan.jpg",
        "role": "Researcher & Developer",
    },
    {
        "name": "Saifullah Saif",
        "email": "saifullahsaif4797@gmail.com",
        "image": "saif.jpg",
        "role": "Researcher",
    },
    {
        "name": "Fysal Sheikh Sheiba",
        "email": "fyshalsheiba@gmail.com",
        "image": "fysal.jpg",
        "role": "Researcher",
    },
]

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SymptoGuard | AI Symptom Checker",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        scroll-behavior: smooth;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: min(96vw, 1500px);
        padding-left: clamp(0.75rem, 2vw, 1.5rem);
        padding-right: clamp(0.75rem, 2vw, 1.5rem);
    }

    section[data-testid="stSidebar"] {
        display: none;
    }

    /* Header */
    .main-header {
        position: relative;
        background: rgba(255, 255, 255, 0.98);
        border: 1px solid #e5e7eb;
        border-radius: 24px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.7rem;
        box-shadow: 0 12px 36px rgba(15, 23, 42, 0.07);
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .brand-icon {
        width: 50px;
        height: 50px;
        border-radius: 16px;
        background: linear-gradient(135deg, #0ea5e9, #14b8a6);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.5rem;
        box-shadow: 0 12px 24px rgba(14, 165, 233, 0.25);
    }

    .brand-title {
        font-size: 1.42rem;
        font-weight: 900;
        color: #0f172a;
        letter-spacing: -0.04em;
        line-height: 1.1;
    }

    .brand-subtitle {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.18rem;
    }

    .nav-links {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        flex-wrap: wrap;
        justify-content: flex-end;
    }

    .nav-links a {
        text-decoration: none;
        color: #334155;
        font-size: 0.88rem;
        font-weight: 800;
        padding: 0.58rem 0.85rem;
        border-radius: 999px;
        transition: 0.2s ease;
    }

    .nav-links a:hover {
        background: #ecfeff;
        color: #0891b2;
    }

    /* Hero */
    .hero {
        background:
            radial-gradient(circle at top left, rgba(255,255,255,0.28), transparent 34%),
            linear-gradient(135deg, #0f172a 0%, #075985 48%, #0f766e 100%);
        color: white;
        border-radius: 36px;
        padding: 3.4rem 3rem;
        margin-bottom: 1.7rem;
        box-shadow: 0 28px 80px rgba(15, 23, 42, 0.22);
        overflow: hidden;
        position: relative;
    }

    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.22);
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 800;
        margin-bottom: 1rem;
    }

    .hero h1 {
        font-size: clamp(2.2rem, 5vw, 4.3rem);
        line-height: 1.02;
        font-weight: 900;
        max-width: 880px;
        margin: 0 0 1rem 0;
        letter-spacing: -0.07em;
    }

    .hero p {
        max-width: 780px;
        color: rgba(255,255,255,0.9);
        font-size: 1.08rem;
        line-height: 1.72;
        margin: 0;
    }

    .hero-stats {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-top: 1.5rem;
    }

    .hero-stat {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 18px;
        padding: 0.82rem 1rem;
        min-width: 150px;
    }

    .hero-stat strong {
        display: block;
        font-size: 1.25rem;
        line-height: 1;
    }

    .hero-stat span {
        font-size: 0.78rem;
        opacity: 0.82;
    }

    /* General Sections */
    .section {
        margin-top: 1.8rem;
        margin-bottom: 1.15rem;
    }

    .section-kicker {
        color: #0891b2;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        font-weight: 900;
        margin-bottom: 0.35rem;
    }

    .section-title {
        color: #0f172a;
        font-size: 2rem;
        font-weight: 900;
        letter-spacing: -0.05em;
        margin: 0 0 0.5rem 0;
    }

    .section-desc {
        color: #64748b;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 820px;
        margin-bottom: 1.1rem;
    }

    .about-research-card {
        background:
            radial-gradient(circle at top right, rgba(124, 58, 237, 0.14), transparent 28%),
            linear-gradient(135deg, #ffffff 0%, #f8fafc 50%, #eef2ff 100%);
        border: 1px solid #c7d2fe;
        border-radius: 28px;
        padding: 1.45rem;
        margin: 0.4rem 0 1.35rem 0;
        box-shadow: 0 18px 48px rgba(79, 70, 229, 0.10);
    }

    .about-research-card h3 {
        color: #312e81;
        font-size: 1.25rem;
        font-weight: 900;
        letter-spacing: -0.035em;
        margin: 0 0 0.55rem 0;
    }

    .about-research-card p {
        color: #475569;
        font-size: 0.96rem;
        line-height: 1.68;
        margin: 0;
    }

    .about-research-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 1rem;
    }

    .about-mini-metric {
        background: rgba(255,255,255,0.78);
        border: 1px solid #e0e7ff;
        border-radius: 18px;
        padding: 0.85rem;
    }

    .about-mini-metric strong {
        display: block;
        color: #4f46e5;
        font-size: 1.08rem;
        font-weight: 900;
        line-height: 1.1;
    }

    .about-mini-metric span {
        color: #64748b;
        font-size: 0.76rem;
        font-weight: 700;
    }

    @media (max-width: 768px) {
        .about-research-grid {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 480px) {
        .about-research-grid {
            grid-template-columns: 1fr;
        }
    }

    /* Step cards */
    .step-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 24px;
        padding: 1.4rem;
        box-shadow: 0 14px 38px rgba(15, 23, 42, 0.06);
        height: 100%;
    }

    .step-number {
        width: 46px;
        height: 46px;
        border-radius: 16px;
        background: #ecfeff;
        color: #0891b2;
        font-weight: 900;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
        border: 1px solid #cffafe;
    }

    .step-card h3 {
        color: #0f172a;
        font-size: 1.06rem;
        font-weight: 900;
        margin: 0 0 0.45rem 0;
    }

    .step-card p {
        color: #64748b;
        font-size: 0.92rem;
        line-height: 1.6;
        margin: 0;
    }

    /* Symptom checker highlight */
    .checker-highlight {
        background:
            radial-gradient(circle at top left, rgba(14, 165, 233, 0.20), transparent 32%),
            linear-gradient(135deg, #ecfeff 0%, #f0fdfa 46%, #ffffff 100%);
        border: 2px solid #67e8f9;
        border-radius: 34px;
        padding: 2rem;
        margin-top: 1.3rem;
        margin-bottom: 1.3rem;
        box-shadow: 0 24px 70px rgba(14, 165, 233, 0.18);
        position: relative;
    }

    .checker-highlight::before {
        content: "START HERE";
        position: absolute;
        top: -14px;
        left: 28px;
        background: linear-gradient(135deg, #0284c7, #0f766e);
        color: white;
        font-size: 0.72rem;
        font-weight: 900;
        letter-spacing: 0.12em;
        padding: 0.42rem 0.8rem;
        border-radius: 999px;
        box-shadow: 0 10px 24px rgba(2, 132, 199, 0.28);
    }

    .checker-highlight h2 {
        margin: 0 0 0.5rem 0;
        color: #0f172a;
        font-size: 2.25rem;
        font-weight: 900;
        letter-spacing: -0.06em;
    }

    .checker-highlight p {
        margin: 0;
        color: #475569;
        font-size: 1rem;
        line-height: 1.65;
        max-width: 860px;
    }

    .big-action-note {
        background: #0f172a;
        color: white;
        border-radius: 20px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.1rem;
        font-weight: 800;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.20);
    }

    .selected-chip {
        display: inline-flex;
        align-items: center;
        background: #ecfeff;
        color: #0e7490;
        border: 1px solid #a5f3fc;
        border-radius: 999px;
        padding: 0.38rem 0.72rem;
        font-size: 0.82rem;
        font-weight: 800;
        margin: 0.18rem;
    }

    /* Result highlight */
    .result-highlight-zone {
        background:
            radial-gradient(circle at top right, rgba(14, 165, 233, 0.16), transparent 30%),
            linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        border: 2px solid #dbeafe;
        border-radius: 34px;
        padding: 1.6rem;
        margin-top: 1.5rem;
        box-shadow: 0 24px 70px rgba(15, 23, 42, 0.10);
    }

    .result-section-title {
        display: flex;
        align-items: center;
        gap: 0.65rem;
        color: #0f172a;
        font-size: 2rem;
        font-weight: 900;
        letter-spacing: -0.05em;
        margin-bottom: 0.8rem;
    }

    .result-section-title span {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 0 8px rgba(34, 197, 94, 0.14);
    }

    .result-ready-banner {
        background: linear-gradient(135deg, #0284c7, #0f766e);
        color: white;
        border-radius: 22px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 18px 42px rgba(2, 132, 199, 0.25);
    }

    .result-ready-banner strong {
        font-size: 1.1rem;
    }

    .result-card {
        background: linear-gradient(180deg, #ffffff 0%, #f0f9ff 100%);
        border: 2px solid #38bdf8;
        border-left: 12px solid #0284c7;
        border-radius: 30px;
        padding: 2rem;
        box-shadow: 0 28px 75px rgba(2, 132, 199, 0.22);
        margin-bottom: 1.2rem;
    }

    .result-label {
        color: #0369a1;
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        font-weight: 900;
        margin-bottom: 0.45rem;
    }

    .result-disease {
        color: #6d28d9;
        background: linear-gradient(135deg, #4c1d95 0%, #7c3aed 42%, #db2777 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: clamp(2.1rem, 4vw, 3rem);
        line-height: 1.1;
        font-weight: 900;
        letter-spacing: -0.06em;
        margin-bottom: 0.75rem;
        text-shadow: 0 10px 28px rgba(124, 58, 237, 0.14);
    }

    .match-badge {
        display: inline-flex;
        background: #0f172a;
        color: white;
        border: 1px solid #0f172a;
        border-radius: 999px;
        padding: 0.52rem 1rem;
        font-size: 0.95rem;
        font-weight: 900;
        margin-bottom: 0.85rem;
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.18);
    }

    .result-message {
        color: #334155;
        font-size: 1.05rem;
        line-height: 1.7;
        margin: 0;
        font-weight: 500;
    }

    .safe-card {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 7px solid #16a34a;
        border-radius: 22px;
        padding: 1rem 1.1rem;
        color: #14532d;
        margin-top: 1rem;
        line-height: 1.55;
    }

    .urgent-card {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        border-left: 7px solid #f97316;
        border-radius: 22px;
        padding: 1rem 1.1rem;
        color: #7c2d12;
        margin-top: 1rem;
        line-height: 1.55;
    }

    .empty-state {
        background: white;
        border: 1px dashed #cbd5e1;
        border-radius: 28px;
        padding: 2.4rem 1.5rem;
        text-align: center;
        color: #64748b;
    }

    .empty-state h3 {
        color: #0f172a;
        font-size: 1.25rem;
        font-weight: 900;
        margin-bottom: 0.35rem;
    }

    /* Team */
    div[data-testid="stImage"] {
        display: flex;
        justify-content: center;
    }

    div[data-testid="stImage"] img {
        border-radius: 999px;
        border: 4px solid #ecfeff;
        box-shadow: 0 10px 28px rgba(14, 165, 233, 0.18);
        object-fit: cover;
        width: 112px !important;
        height: 112px !important;
    }

    .avatar-fallback {
        width: 112px;
        height: 112px;
        border-radius: 999px;
        margin: 0 auto 0.9rem auto;
        background: linear-gradient(135deg, #0ea5e9, #14b8a6);
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.75rem;
        font-weight: 900;
        border: 4px solid #ecfeff;
        box-shadow: 0 10px 28px rgba(14, 165, 233, 0.18);
    }

    .team-name {
        color: #0f172a;
        font-weight: 900;
        font-size: 1rem;
        line-height: 1.25;
        margin-bottom: 0.25rem;
        text-align: center;
    }

    .team-role {
        color: #0891b2;
        font-size: 0.78rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
        text-align: center;
    }

    .team-email {
        color: #64748b;
        font-size: 0.78rem;
        word-break: break-word;
        text-align: center;
    }

    .team-email a {
        color: #0369a1;
        text-decoration: none;
        font-weight: 700;
    }

    .notice {
        background: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 7px solid #f59e0b;
        border-radius: 22px;
        padding: 1rem 1.1rem;
        color: #78350f;
        line-height: 1.58;
        margin-top: 1rem;
    }

    .tech-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }

    .tech-card h4 {
        color: #0f172a;
        font-weight: 900;
        margin-bottom: 0.4rem;
    }

    .footer {
        color: #94a3b8;
        text-align: center;
        font-size: 0.85rem;
        padding: 2rem 0 0.8rem 0;
        line-height: 1.6;
    }

    .stMultiSelect label,
    .stSlider label,
    .stCheckbox label {
        font-weight: 800 !important;
        color: #0f172a !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #06b6d4 0%, #0284c7 42%, #0f766e 100%);
        color: white;
        border: 0;
        border-radius: 999px;
        padding: 1rem 1.25rem;
        font-size: 1.05rem;
        font-weight: 900;
        letter-spacing: -0.01em;
        box-shadow:
            0 18px 42px rgba(2, 132, 199, 0.32),
            inset 0 1px 0 rgba(255,255,255,0.25);
        transition: all 0.22s ease;
        min-height: 3.5rem;
    }

    .stButton > button:hover {
        color: white;
        border: 0;
        transform: translateY(-2px);
        filter: brightness(1.06);
        box-shadow:
            0 24px 55px rgba(2, 132, 199, 0.38),
            inset 0 1px 0 rgba(255,255,255,0.3);
    }

    .stButton > button:active {
        transform: translateY(0px) scale(0.99);
    }

    /* Better spacing and responsiveness */
    @media (min-width: 1500px) {
        .block-container {
            max-width: 1520px;
        }

        .hero {
            padding: 3.8rem 3.4rem;
        }
    }

    @media (max-width: 1100px) {
        .block-container {
            max-width: 98vw;
            padding-left: 0.9rem;
            padding-right: 0.9rem;
        }

        .hero h1 {
            font-size: 3rem;
        }

        .hero {
            padding: 2.6rem 2rem;
        }

        .result-section-title {
            font-size: 1.55rem;
        }

        .checker-highlight h2 {
            font-size: 1.75rem;
        }
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 0.9rem;
            padding-left: 0.65rem;
            padding-right: 0.65rem;
        }

        .main-header {
            padding: 0.85rem;
            border-radius: 18px;
            flex-direction: column;
            align-items: flex-start;
        }

        .brand-icon {
            width: 42px;
            height: 42px;
            font-size: 1.25rem;
        }

        .brand-title {
            font-size: 1.15rem;
        }

        .brand-subtitle {
            font-size: 0.72rem;
        }

        .nav-links {
            gap: 0.2rem;
            justify-content: flex-start;
        }

        .nav-links a {
            font-size: 0.75rem;
            padding: 0.42rem 0.55rem;
        }

        .hero {
            border-radius: 24px;
            padding: 2.1rem 1.2rem;
        }

        .hero h1 {
            font-size: 2.35rem;
            letter-spacing: -0.055em;
        }

        .hero p {
            font-size: 0.95rem;
        }

        .hero-stats {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .hero-stat {
            min-width: 0;
            padding: 0.75rem;
        }

        .checker-highlight {
            border-radius: 24px;
            padding: 1.45rem 1rem;
        }

        .checker-highlight h2 {
            font-size: 1.55rem;
        }

        .big-action-note {
            font-size: 0.88rem;
            border-radius: 16px;
        }

        .result-highlight-zone {
            border-radius: 24px;
            padding: 1.1rem;
        }

        .result-card {
            border-radius: 24px;
            padding: 1.35rem;
        }

        .result-disease {
            font-size: 2rem;
        }

        .stButton > button {
            width: 100%;
            min-height: 3.3rem;
            font-size: 0.98rem;
        }
    }

    @media (max-width: 480px) {
        .hero h1 {
            font-size: 2rem;
        }

        .hero-stats {
            grid-template-columns: 1fr;
        }

        .section-title {
            font-size: 1.55rem;
        }

        .checker-highlight h2 {
            font-size: 1.35rem;
        }

        .result-section-title {
            font-size: 1.25rem;
        }

        .result-disease {
            font-size: 1.65rem;
        }
    }


    /* ─────────────────────────────────────────────
       Final responsive fixes
       - Keeps header visible above content
       - Prevents graphs from being cropped into circles
       - Adds fully responsive navigation for phone and PC
    ───────────────────────────────────────────── */
    .main-header {
        position: sticky !important;
        top: 0.6rem !important;
        z-index: 1000 !important;
        width: 100% !important;
        box-sizing: border-box !important;
        overflow: visible !important;
        background: rgba(255, 255, 255, 0.96) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
    }

    .main-header,
    .main-header * {
        box-sizing: border-box;
    }

    .brand {
        min-width: 0;
        flex: 1 1 auto;
    }

    .brand-icon {
        flex: 0 0 auto;
    }

    .brand > div:last-child {
        min-width: 0;
        overflow: hidden;
    }

    .brand-title,
    .brand-subtitle {
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .nav-links {
        min-width: 0;
        flex: 0 1 auto;
    }

    .nav-links a {
        white-space: nowrap;
    }

    /* Reset Streamlit image styling so charts/graphs show as full images, not circular thumbnails. */
    div[data-testid="stImage"] img {
        border-radius: 0 !important;
        border: 0 !important;
        box-shadow: none !important;
        object-fit: contain !important;
        width: auto !important;
        height: auto !important;
        max-width: 100% !important;
    }

    .team-photo {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0 auto 0.9rem auto;
    }

    .team-avatar-img {
        width: 112px;
        height: 112px;
        border-radius: 999px;
        border: 4px solid #ecfeff;
        box-shadow: 0 10px 28px rgba(14, 165, 233, 0.18);
        object-fit: cover;
        display: block;
    }

    @media (max-width: 1180px) {
        .main-header {
            align-items: stretch !important;
            gap: 0.85rem !important;
        }

        .nav-links a {
            font-size: 0.82rem !important;
            padding: 0.52rem 0.68rem !important;
        }
    }

    @media (max-width: 900px) {
        .block-container {
            max-width: 100vw !important;
            padding-left: 0.65rem !important;
            padding-right: 0.65rem !important;
        }

        .main-header {
            top: 0.35rem !important;
            border-radius: 20px !important;
            padding: 0.85rem !important;
            flex-direction: column !important;
            align-items: stretch !important;
        }

        .brand {
            width: 100%;
            gap: 0.7rem;
        }

        .brand-title {
            font-size: clamp(1rem, 4vw, 1.2rem) !important;
        }

        .brand-subtitle {
            font-size: clamp(0.68rem, 3vw, 0.78rem) !important;
        }

        .nav-links {
            width: 100% !important;
            display: grid !important;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.35rem !important;
        }

        .nav-links a {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 2.35rem;
            padding: 0.5rem 0.35rem !important;
            border-radius: 14px !important;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            font-size: clamp(0.64rem, 2.55vw, 0.78rem) !important;
            line-height: 1.15;
            text-align: center;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .hero {
            margin-top: 1rem;
            border-radius: 26px !important;
            padding: clamp(1.35rem, 5vw, 2.2rem) !important;
        }

        .hero-badge {
            max-width: 100%;
            white-space: normal;
            line-height: 1.35;
        }

        .hero h1 {
            font-size: clamp(2rem, 9vw, 3rem) !important;
            letter-spacing: -0.055em !important;
            word-break: normal;
        }

        .hero p {
            font-size: clamp(0.92rem, 3.6vw, 1.02rem) !important;
        }
    }

    @media (max-width: 520px) {
        .main-header {
            padding: 0.75rem !important;
            border-radius: 18px !important;
        }

        .brand-icon {
            width: 44px !important;
            height: 44px !important;
            border-radius: 14px !important;
            font-size: 1.2rem !important;
        }

        .nav-links {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .nav-links a {
            font-size: 0.76rem !important;
            padding: 0.5rem 0.4rem !important;
        }

        .hero-stats {
            grid-template-columns: 1fr !important;
        }

        .hero-stat {
            width: 100%;
        }

        .result-card,
        .result-highlight-zone,
        .checker-highlight,
        .step-card,
        .about-research-card {
            border-radius: 20px !important;
        }
    }


    /* ─────────────────────────────────────────────
       FINAL DEVICE FIXES - header + charts
       This block intentionally comes last so it overrides older rules.
    ───────────────────────────────────────────── */
    header[data-testid="stHeader"] {
        background: transparent !important;
        box-shadow: none !important;
    }

    .block-container,
    div[data-testid="stVerticalBlock"],
    div[data-testid="stVerticalBlock"] > div,
    div.element-container,
    div[data-testid="stMarkdownContainer"] {
        overflow: visible !important;
    }

    .sticky-header-shell {
        position: sticky !important;
        top: 0 !important;
        z-index: 1000 !important;
        width: 100% !important;
        padding-top: max(0.65rem, env(safe-area-inset-top)) !important;
        padding-bottom: 0.35rem !important;
        margin-bottom: 1.35rem !important;
        background: linear-gradient(180deg, rgba(248,250,252,0.98) 0%, rgba(248,250,252,0.82) 72%, rgba(248,250,252,0) 100%) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        box-sizing: border-box !important;
    }

    .sticky-header-shell .main-header {
        position: relative !important;
        top: auto !important;
        z-index: 1000 !important;
        width: 100% !important;
        min-height: 78px !important;
        margin: 0 !important;
        padding: 0.9rem 1.15rem !important;
        display: grid !important;
        grid-template-columns: minmax(220px, 1fr) auto !important;
        align-items: center !important;
        gap: 0.9rem !important;
        overflow: visible !important;
        border-radius: 24px !important;
        background: rgba(255,255,255,0.97) !important;
        border: 1px solid rgba(226,232,240,0.95) !important;
        box-shadow: 0 16px 45px rgba(15,23,42,0.10) !important;
        box-sizing: border-box !important;
    }

    .sticky-header-shell .brand {
        display: grid !important;
        grid-template-columns: 56px minmax(0, 1fr) !important;
        align-items: center !important;
        gap: 0.78rem !important;
        min-width: 0 !important;
        width: 100% !important;
        overflow: visible !important;
    }

    .sticky-header-shell .brand-icon {
        width: 56px !important;
        height: 56px !important;
        min-width: 56px !important;
        min-height: 56px !important;
        border-radius: 16px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        line-height: 1 !important;
        overflow: visible !important;
    }

    .sticky-header-shell .brand-title {
        font-size: clamp(1.08rem, 1.8vw, 1.42rem) !important;
        line-height: 1.18 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }

    .sticky-header-shell .brand-subtitle {
        font-size: clamp(0.72rem, 1.1vw, 0.82rem) !important;
        line-height: 1.3 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }

    .sticky-header-shell .nav-links {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-end !important;
        gap: 0.35rem !important;
        flex-wrap: wrap !important;
        width: auto !important;
        min-width: 0 !important;
        overflow: visible !important;
    }

    .sticky-header-shell .nav-links a {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        min-height: 38px !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: clip !important;
        line-height: 1.15 !important;
    }

    /* Graphs must remain rectangular/full-width. This overrides the old global circular image rule. */
    div[data-testid="stImage"],
    div[data-testid="stImage"] > div,
    div[data-testid="stImage"] img,
    div[data-testid="stPyplot"],
    div[data-testid="stPyplot"] img,
    div[data-testid="stPyplot"] canvas,
    .stImage img,
    .stPlotlyChart img,
    .element-container img:not(.team-avatar-img) {
        border-radius: 0 !important;
        border: 0 !important;
        box-shadow: none !important;
        clip-path: none !important;
        object-fit: contain !important;
        max-width: 100% !important;
        height: auto !important;
        overflow: visible !important;
    }

    .team-avatar-img,
    .avatar-fallback {
        border-radius: 999px !important;
        width: 112px !important;
        height: 112px !important;
        min-width: 112px !important;
        min-height: 112px !important;
        object-fit: cover !important;
    }

    @media (max-width: 980px) {
        .sticky-header-shell {
            padding-top: max(0.5rem, env(safe-area-inset-top)) !important;
            margin-bottom: 1rem !important;
        }

        .sticky-header-shell .main-header {
            grid-template-columns: 1fr !important;
            min-height: auto !important;
            padding: 0.85rem !important;
            gap: 0.75rem !important;
            border-radius: 22px !important;
        }

        .sticky-header-shell .brand {
            grid-template-columns: 52px minmax(0, 1fr) !important;
            min-height: 56px !important;
        }

        .sticky-header-shell .brand-icon {
            width: 52px !important;
            height: 52px !important;
            min-width: 52px !important;
            min-height: 52px !important;
        }

        .sticky-header-shell .nav-links {
            display: grid !important;
            grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
            width: 100% !important;
            gap: 0.4rem !important;
        }

        .sticky-header-shell .nav-links a {
            width: 100% !important;
            min-height: 40px !important;
            padding: 0.52rem 0.35rem !important;
            border-radius: 14px !important;
            background: #f8fafc !important;
            border: 1px solid #e2e8f0 !important;
            font-size: clamp(0.66rem, 2.35vw, 0.78rem) !important;
            white-space: normal !important;
            text-align: center !important;
        }

        .hero {
            margin-top: 0.8rem !important;
        }
    }

    @media (max-width: 520px) {
        .block-container {
            padding-left: 0.55rem !important;
            padding-right: 0.55rem !important;
            padding-top: 0.25rem !important;
        }

        .sticky-header-shell .main-header {
            padding: 0.75rem !important;
            border-radius: 20px !important;
        }

        .sticky-header-shell .brand {
            grid-template-columns: 48px minmax(0, 1fr) !important;
            gap: 0.65rem !important;
            min-height: 54px !important;
        }

        .sticky-header-shell .brand-icon {
            width: 48px !important;
            height: 48px !important;
            min-width: 48px !important;
            min-height: 48px !important;
        }

        .sticky-header-shell .brand-title {
            font-size: 1.1rem !important;
        }

        .sticky-header-shell .brand-subtitle {
            font-size: 0.76rem !important;
        }

        .sticky-header-shell .nav-links {
            grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
        }

        .sticky-header-shell .nav-links a {
            min-height: 38px !important;
            font-size: 0.74rem !important;
        }

        .hero h1 {
            font-size: clamp(1.85rem, 8.5vw, 2.45rem) !important;
        }
    }


    /* ─────────────────────────────────────────────
       Clean professional CTA button
       Keep this block at the very end so it overrides earlier button CSS.
    ───────────────────────────────────────────── */
    .cta-hint {
        text-align: center;
        color: #334155;
        font-size: 0.95rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
        letter-spacing: -0.01em;
    }

    /* This targets the Streamlit primary action button. */
    .stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #0f766e 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(15, 118, 110, 0.35) !important;
        border-radius: 14px !important;
        padding: 1.05rem 1.35rem !important;
        font-size: 1.12rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.01em !important;
        min-height: 4.05rem !important;
        width: 100% !important;
        box-shadow: 0 12px 28px rgba(2, 132, 199, 0.24) !important;
        transform: none !important;
        transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease !important;
        animation: none !important;
    }

    .stButton > button:hover {
        color: #ffffff !important;
        border: 1px solid rgba(15, 118, 110, 0.45) !important;
        transform: translateY(-2px) !important;
        filter: brightness(1.04) !important;
        box-shadow: 0 16px 34px rgba(2, 132, 199, 0.30) !important;
    }

    .stButton > button:active {
        transform: translateY(0) scale(0.99) !important;
        box-shadow: 0 10px 22px rgba(2, 132, 199, 0.22) !important;
    }

    .stButton > button:focus {
        outline: 3px solid rgba(14, 165, 233, 0.22) !important;
        outline-offset: 3px !important;
    }

    @media (max-width: 768px) {
        .stButton > button {
            font-size: 1rem !important;
            min-height: 3.8rem !important;
            padding: 0.95rem 1.05rem !important;
            border-radius: 13px !important;
        }

        .cta-hint {
            font-size: 0.86rem !important;
        }
    }

</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def clean_label(text: str) -> str:
    return str(text).replace("_", " ").replace("-", " ").strip().title()


def confidence_label(probability: float):
    percent = probability * 100

    if percent >= 70:
        return "High match", "Your selected symptoms strongly match this condition."
    if percent >= 40:
        return "Moderate match", "Your selected symptoms partially match this condition."
    if percent >= 20:
        return (
            "Low to moderate match",
            "Some selected symptoms match this condition, but similar symptoms may also appear in other conditions.",
        )
    return (
        "Low match",
        "The system found a weak match. Review the selected symptoms and consult a qualified doctor if symptoms continue.",
    )


def build_display_maps(symptom_list):
    raw_to_display = {}
    display_to_raw = {}
    used_names = set()

    for raw in symptom_list:
        display = clean_label(raw)
        final_display = display

        if final_display in used_names:
            final_display = f"{display} [{raw}]"

        used_names.add(final_display)
        raw_to_display[raw] = final_display
        display_to_raw[final_display] = raw

    return raw_to_display, display_to_raw


def decode_class_from_position(model_obj, label_encoder, position: int) -> str:
    if hasattr(model_obj, "classes_"):
        model_class = model_obj.classes_[position]

        try:
            if isinstance(model_class, (int, np.integer)):
                return str(label_encoder.inverse_transform([int(model_class)])[0])
            return str(model_class)
        except Exception:
            try:
                return str(label_encoder.classes_[int(model_class)])
            except Exception:
                return str(label_encoder.classes_[position])

    return str(label_encoder.classes_[position])


def make_top_conditions_chart(top_conditions):
    names = [clean_label(disease) for disease, _ in top_conditions]
    values = [probability * 100 for _, probability in top_conditions]

    fig, ax = plt.subplots(figsize=(7.2, max(3.3, len(names) * 0.62)))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    colors = ["#0284c7" if idx == 0 else "#bae6fd" for idx in range(len(values))]

    bars = ax.barh(names[::-1], values[::-1], color=colors[::-1], height=0.58)
    ax.set_xlabel("Match level (%)", fontsize=9)
    ax.set_xlim(0, max(max(values) * 1.25, 5))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", alpha=0.15)

    for bar, value in zip(bars, values[::-1]):
        ax.text(
            bar.get_width() + 0.35,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#334155",
        )

    fig.tight_layout()
    return fig


def make_symptom_impact_chart(symptom_names, shap_values, disease_name):
    paired = sorted(zip(symptom_names, shap_values), key=lambda item: item[1])
    names = [item[0] for item in paired]
    values = [item[1] for item in paired]
    colors = ["#16a34a" if value >= 0 else "#dc2626" for value in values]

    fig, ax = plt.subplots(figsize=(7.2, max(3.3, len(names) * 0.6)))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    bars = ax.barh(names, values, color=colors, height=0.58)

    ax.axvline(0, color="#334155", linewidth=0.85, linestyle="--")
    ax.set_xlabel("Influence on this result", fontsize=9)
    ax.set_title(
        f"Symptom Influence → {clean_label(disease_name)}",
        fontsize=10,
        fontweight="bold",
        pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="x", alpha=0.14)

    for bar, value in zip(bars, values):
        x_pos = bar.get_width()
        ax.text(
            x_pos + (0.001 if x_pos >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f}",
            va="center",
            ha="left" if x_pos >= 0 else "right",
            fontsize=7.8,
            color="#334155",
        )

    fig.tight_layout()
    return fig


def get_initials(name: str) -> str:
    parts = [part for part in name.split() if part]
    if len(parts) == 1:
        return parts[0][:2].upper()
    return "".join(part[0] for part in parts[:2]).upper()


def get_team_image_path(image_name: str):
    image_path = ASSETS_DIR / image_name
    return image_path if image_path.exists() else None


def image_to_data_uri(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    encoded_image = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded_image}"


def render_team_member(member: dict):
    name = member["name"]
    email = member["email"]
    role = member["role"]
    image_name = member["image"]
    image_path = get_team_image_path(image_name)

    with st.container(border=True):
        if image_path:
            image_src = html.escape(image_to_data_uri(image_path), quote=True)
            image_alt = html.escape(name, quote=True)
            st.markdown(
                f'<div class="team-photo"><img class="team-avatar-img" src="{image_src}" alt="{image_alt}"></div>',
                unsafe_allow_html=True,
            )
        else:
            initials = html.escape(get_initials(name))
            st.markdown(
                f'<div class="avatar-fallback">{initials}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="team-name">{html.escape(name)}</div>
            <div class="team-role">{html.escape(role)}</div>
            <div class="team-email">
                <a href="mailto:{html.escape(email)}">{html.escape(email)}</a>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# LOAD MODEL + DATA
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading SymptoGuard...")
def load_resources():
    required_files = [MODEL_PATH, LE_PATH, FEATURES_PATH, DATA_PATH]
    missing_files = [path for path in required_files if not path.exists()]

    if missing_files:
        missing_text = "\n".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Missing required file(s):\n{missing_text}")

    loaded_model = joblib.load(MODEL_PATH)
    loaded_label_encoder = joblib.load(LE_PATH)
    loaded_symptoms = list(joblib.load(FEATURES_PATH))

    df_raw = pd.read_csv(DATA_PATH)

    # The updated 70/15/15 model was trained with cleaned feature names.
    # Example: "shortness of breath" -> "shortness_of_breath".
    df_raw.columns = df_raw.columns.str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)

    if "diseases" not in df_raw.columns:
        raise ValueError("The dataset must contain a 'diseases' column.")

    missing_features = [feature for feature in loaded_symptoms if feature not in df_raw.columns]
    if missing_features:
        raise ValueError(
            "The dataset is missing feature columns expected by the model:\n"
            + "\n".join(missing_features[:30])
        )

    sample_size = min(200, len(df_raw))
    background = df_raw[loaded_symptoms].sample(n=sample_size, random_state=42).values

    return loaded_model, loaded_label_encoder, loaded_symptoms, background


try:
    model, label_encoder, symptoms, X_background = load_resources()
except Exception as error:
    st.error("SymptoGuard could not start.")
    st.code(str(error))
    st.info("Make sure models/best_nb_model.pkl, models/label_encoder.pkl, models/feature_columns.pkl, and data/100_Disease.csv are inside the GitHub project folder.")
    st.stop()


@st.cache_resource(show_spinner="Preparing explanation engine...")
def get_explainer(_model, _background_values, _symptoms_tuple):
    feature_names = list(_symptoms_tuple)

    def predict_from_array(values):
        values = np.asarray(values)

        if values.ndim == 1:
            values = values.reshape(1, -1)

        values_df = pd.DataFrame(values, columns=feature_names)
        return _model.predict_proba(values_df)

    return shap.PermutationExplainer(
        predict_from_array,
        _background_values[:50],
        max_evals=2 * len(feature_names) + 1,
    )


raw_to_display, display_to_raw = build_display_maps(symptoms)
display_options = sorted(display_to_raw.keys())

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="sticky-header-shell">
    <div class="main-header">
        <div class="brand">
            <div class="brand-icon">🩺</div>
            <div>
                <div class="brand-title">SymptoGuard</div>
                <div class="brand-subtitle">AI-assisted symptom checker</div>
            </div>
        </div>
        <div class="nav-links">
            <a href="#home">Home</a>
            <a href="#how-it-works">How It Works</a>
            <a href="#symptom-checker">Symptom Checker</a>
            <a href="#about-us">About Us</a>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown('<div id="home"></div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="hero">
    <div class="hero-badge">Preliminary health screening · Built for easy use</div>
    <h1>Understand possible health conditions from symptoms.</h1>
    <p>
        SymptoGuard helps users select symptoms, view possible condition matches, and understand
        which symptoms influenced the result. It is designed for simple, non-technical use.
    </p>
    <div class="hero-stats">
        <div class="hero-stat"><strong>100</strong><span>Possible disease classes</span></div>
        <div class="hero-stat"><strong>227</strong><span>Symptoms supported</span></div>
        <div class="hero-stat"><strong>101,903</strong><span>Training records used</span></div>
        <div class="hero-stat"><strong>89.60%</strong><span>Final test accuracy</span></div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# HOW IT WORKS
# ─────────────────────────────────────────────
st.markdown('<div id="how-it-works"></div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="section">
    <div class="section-kicker">How it works</div>
    <div class="section-title">A simple 3-step health assistant</div>
    <div class="section-desc">
        The website is structured so a non-technical person can use it without understanding machine learning.
    </div>
</div>
""",
    unsafe_allow_html=True,
)

step_col_1, step_col_2, step_col_3 = st.columns(3)

with step_col_1:
    st.markdown(
        """
<div class="step-card">
    <div class="step-number">01</div>
    <h3>Choose symptoms</h3>
    <p>Search and select the symptoms the patient is currently experiencing.</p>
</div>
""",
        unsafe_allow_html=True,
    )

with step_col_2:
    st.markdown(
        """
<div class="step-card">
    <div class="step-number">02</div>
    <h3>Check possible conditions</h3>
    <p>The system compares the selected symptoms with learned disease patterns.</p>
</div>
""",
        unsafe_allow_html=True,
    )

with step_col_3:
    st.markdown(
        """
<div class="step-card">
    <div class="step-number">03</div>
    <h3>Understand the result</h3>
    <p>The explanation shows which selected symptoms influenced the result most.</p>
</div>
""",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# SYMPTOM CHECKER
# ─────────────────────────────────────────────
st.markdown('<div id="symptom-checker"></div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="checker-highlight">
    <div class="section-kicker">Symptom Checker</div>
    <h2>Search symptoms and check possible conditions</h2>
    <p>
        Start here. Type symptoms in the search box, select everything the patient is experiencing,
        then click <strong>Check Possible Conditions</strong>.
    </p>
</div>
""",
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown(
        """
<div class="big-action-note">
    Step 1: Select symptoms below. Step 2: Click the button. Step 3: Review the result and explanation.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("#### Select symptoms")

    selected_display = st.multiselect(
        "Search and select symptoms",
        options=display_options,
        placeholder="Example: fever, cough, headache, acne...",
        label_visibility="collapsed",
    )

    selected_symptoms = [display_to_raw[item] for item in selected_display]

    if selected_symptoms:
        selected_html = "".join(
            f'<span class="selected-chip">{html.escape(clean_label(symptom))}</span>'
            for symptom in selected_symptoms
        )
        st.markdown(
            f"""
            <div style="margin-top:0.8rem; margin-bottom:1rem;">
                <div style="font-weight:900; color:#0f172a; margin-bottom:0.35rem;">
                    Selected symptoms: {len(selected_symptoms)}
                </div>
                {selected_html}
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("No symptoms selected yet.")

    control_col_1, control_col_2, control_col_3 = st.columns([0.85, 0.85, 1.65])

    with control_col_1:
        top_n = st.slider(
            "Possible conditions to show",
            min_value=3,
            max_value=10,
            value=5,
        )

    with control_col_2:
        show_explanation = st.checkbox(
            "Show symptom explanation",
            value=True,
        )

    with control_col_3:
        
        predict_btn = st.button(
            "Check Possible Conditions  →",
            type="primary",
            use_container_width=True,
        )

# ─────────────────────────────────────────────
# PREDICTION LOGIC
# ─────────────────────────────────────────────
if predict_btn:
    if not selected_symptoms:
        st.warning("Please select at least one symptom before checking possible conditions.")
        st.session_state["predicted"] = False
    else:
        input_vector = np.zeros(len(symptoms), dtype=int)

        for symptom in selected_symptoms:
            if symptom in symptoms:
                input_vector[symptoms.index(symptom)] = 1

        input_df = pd.DataFrame([input_vector], columns=symptoms)

        try:
            probabilities = model.predict_proba(input_df)[0]
        except Exception as error:
            st.error("Prediction failed.")
            st.code(str(error))
            st.stop()

        top_positions = np.argsort(probabilities)[::-1][:top_n]

        top_conditions = [
            (
                decode_class_from_position(model, label_encoder, int(position)),
                float(probabilities[position]),
            )
            for position in top_positions
        ]

        best_disease, best_probability = top_conditions[0]
        best_position = int(top_positions[0])

        st.session_state["predicted"] = True
        st.session_state["input_array"] = input_vector.reshape(1, -1)
        st.session_state["selected_symptoms"] = selected_symptoms
        st.session_state["top_conditions"] = top_conditions
        st.session_state["best_disease"] = best_disease
        st.session_state["best_probability"] = best_probability
        st.session_state["best_position"] = best_position

# ─────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────
result_col, explanation_col = st.columns([1.05, 0.95], gap="large")

with result_col:
    st.markdown(
        """
<div class="result-highlight-zone">
    <div class="result-section-title"><span></span> Result: Possible condition matches</div>
    <p style="color:#64748b; font-size:1rem; line-height:1.6; margin-top:-0.2rem;">
        After selecting symptoms, the most likely condition and other possible matches will appear here.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    if st.session_state.get("predicted"):
        best_disease = st.session_state["best_disease"]
        best_probability = st.session_state["best_probability"]
        top_conditions = st.session_state["top_conditions"]

        match_title, match_message = confidence_label(best_probability)

        st.markdown(
            """
<div class="result-ready-banner">
    <strong>Result ready.</strong><br>
    Review the most likely condition, other possible matches, and symptom explanation below.
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
<div class="result-card">
    <div class="result-label">Most likely condition</div>
    <div class="result-disease">{html.escape(clean_label(best_disease))}</div>
    <div class="match-badge">{html.escape(match_title)} · {best_probability * 100:.1f}%</div>
    <p class="result-message">{html.escape(match_message)}</p>
</div>
""",
            unsafe_allow_html=True,
        )

        with st.container(border=True):
            st.markdown("#### Other possible conditions")
            st.caption("These are possible matches, not confirmed diagnoses.")

            fig = make_top_conditions_chart(top_conditions)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        st.markdown(
            """
<div class="safe-card">
    <strong>Recommended next step:</strong><br>
    If symptoms are painful, spreading, worsening, persistent, or worrying, consult a qualified doctor.
    Use this result as a starting point for medical discussion, not as a final diagnosis.
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="urgent-card">
    <strong>Seek urgent medical help immediately if there is:</strong><br>
    Chest pain, breathing difficulty, fainting, confusion, severe fever, uncontrolled bleeding,
    severe allergic reaction, or rapidly worsening symptoms.
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
<div class="empty-state">
    <h3>No result yet</h3>
    <p>Select symptoms above, then click <strong>Check Possible Conditions</strong>.</p>
</div>
""",
            unsafe_allow_html=True,
        )

with explanation_col:
    st.markdown(
        """
<div class="result-highlight-zone">
    <div class="result-section-title"><span></span> Explanation</div>
    <p style="color:#64748b; font-size:1rem; line-height:1.6; margin-top:-0.2rem;">
        This section shows which selected symptoms influenced the prediction most.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    if not show_explanation:
        st.info("Turn on 'Show symptom explanation' to see why the result appeared.")
    elif st.session_state.get("predicted"):
        input_array = st.session_state["input_array"]
        best_disease = st.session_state["best_disease"]
        best_position = st.session_state["best_position"]

        with st.spinner("Preparing symptom explanation..."):
            try:
                explainer = get_explainer(model, X_background, tuple(symptoms))
                shap_values = explainer(input_array)
                raw_values = shap_values.values

                if raw_values.ndim == 3:
                    symptom_values = raw_values[0, :, best_position]
                elif raw_values.ndim == 2:
                    if raw_values.shape[1] == len(label_encoder.classes_):
                        symptom_values = raw_values[:, best_position]
                    else:
                        symptom_values = raw_values[0, :]
                else:
                    symptom_values = raw_values

                active_indices = np.where(input_array[0] == 1)[0]

                active_impacts = []
                for idx in active_indices:
                    if idx < len(symptom_values):
                        active_impacts.append((symptoms[idx], float(symptom_values[idx])))

                active_impacts.sort(key=lambda item: abs(item[1]), reverse=True)
                top_active = active_impacts[:12]

                if not top_active:
                    st.info("No selected symptom explanation available.")
                else:
                    impact_names = [clean_label(symptom) for symptom, _ in top_active]
                    impact_values = [value for _, value in top_active]

                    with st.container(border=True):
                        fig2 = make_symptom_impact_chart(
                            impact_names,
                            impact_values,
                            best_disease,
                        )
                        st.pyplot(fig2, use_container_width=True)
                        plt.close(fig2)

                        positive_drivers = [
                            (name, value)
                            for name, value in zip(impact_names, impact_values)
                            if value > 0
                        ]

                        if positive_drivers:
                            main_driver = max(positive_drivers, key=lambda item: item[1])
                            st.markdown(
                                f"""
                                <p style="color:#334155; line-height:1.6;">
                                    <strong>Main reason:</strong>
                                    <strong>{html.escape(main_driver[0])}</strong> had the strongest positive influence
                                    on the result for <strong>{html.escape(clean_label(best_disease))}</strong>.
                                </p>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                """
                                <p style="color:#334155; line-height:1.6;">
                                    The selected symptoms did not strongly support one single condition.
                                    Review symptom selection or consult a doctor.
                                </p>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.caption(
                            "Green bars support this result. Red bars reduce support. Longer bars mean stronger influence."
                        )

            except Exception as error:
                st.error("The explanation could not be generated.")
                st.code(str(error))
                st.info("The prediction is still usable. Try selecting fewer symptoms and run again.")
    else:
        st.markdown(
            """
<div class="empty-state">
    <h3>No explanation yet</h3>
    <p>After checking symptoms, this section will explain which symptoms influenced the result.</p>
</div>
""",
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# ABOUT US
# ─────────────────────────────────────────────
st.markdown('<div id="about-us"></div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="section">
    <div class="section-kicker">About Us</div>
    <div class="section-title">Meet the SymptoGuard team</div>
    <div class="section-desc">
        SymptoGuard was developed by Computer Science and Engineering students from
        American International University-Bangladesh as an academic AI health-assistance project.
        The project focuses on symptom-based disease prediction, transparent explanations,
        and a simple web experience for preliminary health awareness.
    </div>
</div>

<div class="about-research-card">
    <h3>Research background behind SymptoGuard</h3>
    <p>
        This system is based on a symptom-driven machine learning framework for multiclass disease diagnosis.
        The study used a large Diseases and Symptoms dataset, converted symptoms into binary present/absent
        features, used a 70/15/15 train-validation-test split, compared multiple machine learning
        and ensemble models, and selected the tuned Bernoulli Naive Bayes model for deployment. SHAP explainability is included so users can understand which
        selected symptoms contributed most to a possible condition match.
    </p>
    <div class="about-research-grid">
        <div class="about-mini-metric"><strong>101,903</strong><span>processed patient records</span></div>
        <div class="about-mini-metric"><strong>227</strong><span>binary symptom features</span></div>
        <div class="about-mini-metric"><strong>100</strong><span>disease classes</span></div>
        <div class="about-mini-metric"><strong>89.60%</strong><span>final test accuracy</span></div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

team_cols = st.columns(4, gap="medium")

for col, member in zip(team_cols, TEAM_MEMBERS):
    with col:
        render_team_member(member)

# ─────────────────────────────────────────────
# SAFETY + TECHNICAL INFO
# ─────────────────────────────────────────────
st.markdown("---")

info_tab, safety_tab, technical_tab = st.tabs(
    ["About SymptoGuard", "Medical Safety", "Technical Details"]
)

with info_tab:
    st.markdown(
        """
        SymptoGuard is an AI-assisted symptom checker designed for preliminary health awareness.
        Users select symptoms, and the system returns possible condition matches with a simple explanation.

        The research workflow behind the app includes dataset cleaning, zero-variance feature removal,
        label encoding, stratified train-test splitting, model comparison, Optuna-based hyperparameter
        tuning, and SHAP-based explainability. The goal is to support early awareness and clinical
        discussion by showing both a ranked condition match and symptom-level reasoning.
        """
    )

with safety_tab:
    st.markdown(
        """
<div class="notice">
    <strong>Important medical notice:</strong><br>
    SymptoGuard is not a doctor and cannot diagnose, treat, or prescribe medicine.
    Symptoms can overlap across many diseases, and the result may be incomplete or incorrect.
    Always consult a qualified healthcare professional for proper diagnosis and treatment.
</div>
""",
        unsafe_allow_html=True,
    )

with technical_tab:
    st.markdown(
        """
<div class="tech-card">
    <h4>Dataset Summary</h4>
    <ul>
        <li><strong>Records:</strong> 101,903 patient/symptom records</li>
        <li><strong>Symptom features:</strong> 227 binary symptom features</li>
        <li><strong>Disease classes:</strong> 100 multiclass disease categories</li>
        <li><strong>Input type:</strong> symptom presence/absence encoded as binary values</li>
        <li><strong>Data split:</strong> 70% training, 15% validation, 15% final test</li>
    </ul>
</div>

<div class="tech-card">
    <h4>Machine Learning Models Evaluated</h4>
    <ul>
        <li>Logistic Regression</li>
        <li>Bernoulli Naive Bayes</li>
        <li>k-Nearest Neighbors</li>
        <li>XGBoost</li>
        <li>CatBoost-based soft voting ensemble</li>
        <li>Soft Voting Ensemble</li>
        <li>Stacking Ensemble</li>
    </ul>
</div>

<div class="tech-card">
    <h4>Final Deployed Model</h4>
    <ul>
        <li><strong>Model:</strong> Tuned Bernoulli Naive Bayes</li>
        <li><strong>Split:</strong> 70% training, 15% validation, 15% final test</li>
        <li><strong>Optimization:</strong> Optuna hyperparameter tuning</li>
        <li><strong>Best alpha:</strong> 0.3497</li>
        <li><strong>Final test accuracy:</strong> 89.60%</li>
        <li><strong>Final test weighted F1-score:</strong> 89.66%</li>
        <li><strong>Final test macro F1-score:</strong> 89.62%</li>
        <li><strong>Explainability:</strong> SHAP-based symptom contribution analysis</li>
    </ul>
</div>
""",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="footer">
    SymptoGuard · AI-assisted symptom checker · Academic research prototype<br>
    Built for preliminary screening, health awareness, and thesis demonstration.
</div>
""",
    unsafe_allow_html=True,
)
