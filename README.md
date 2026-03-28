# ◇ ML Churn Predictor — Customer Retention Intelligence

Predict which customers will leave. Know why. Act before they do.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)

## What it does

Upload any customer dataset (CSV/Excel) → the tool automatically trains ML models, identifies churn risk, and delivers actionable retention strategies.

**Full ML pipeline in one click:**
- Automatic feature detection and encoding
- Model training: Random Forest vs Logistic Regression (with comparison)
- Confusion Matrix and ROC Curve visualization
- Feature importance analysis with partial dependence plots
- Customer risk segmentation (High / Medium / Low)
- Revenue impact analysis
- Segment-level churn breakdown
- Exportable Excel report with findings and recommendations

## Business value

| Metric | What it shows |
|--------|--------------|
| Churn probability per customer | Who is about to leave |
| Feature importance | Why they leave |
| Partial dependence | At what threshold they leave |
| Revenue at risk | How much money is at stake |
| Segment analysis | Which groups churn most |
| Recommendations | What to do about it |

## Quick start

```bash
git clone https://github.com/YOUR_USERNAME/ml-churn-predictor.git
cd ml-churn-predictor
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free on Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → Deploy

## Tech stack

- **Streamlit** — Interactive web interface
- **scikit-learn** — Random Forest, Logistic Regression, metrics
- **Pandas / NumPy** — Data processing
- **Plotly** — Interactive charts (confusion matrix, ROC, partial dependence)

## Demo dataset

Built-in sample: 1,000 SaaS customers with realistic churn correlations. Features include tenure, monthly spend, satisfaction score, support tickets, contract type, login frequency, and more.

## Author

**[YOUR NAME]** — ML Engineer & Data Analyst
- Physics research background with hands-on ML experience
- Based in Luxembourg (EN/FR/DE)
- Available for freelance projects
