# Churn Risk Analysis and Customer Lifetime Value (CLV) Prediction

This repository contains the code for analyzing customer churn risk using Accelerated Failure Time (AFT) models (Log-Logistic, Log-Normal, Weibull) and calculating Customer Lifetime Value (CLV) across different customer segments. The project includes:

- Survival model fitting and comparison,
- CLV prediction per customer,
- Segment analysis and visualization.

## 📂 Repository Structure

- `aft.py`: Contains functions to fit AFT models, plot survival curves, and calculate CLV.
- `main.ipynb`: Jupyter Notebook demonstrating step-by-step usage of the functions and providing analysis results.
- `requirements.txt`: List of Python dependencies.
- `README.md`: This file.

## 🚀 Getting Started

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/churn-clv-analysis.git
cd churn-clv-analysis

### 2️⃣ Set up the environment

```bash
pip install -r requirements.txt

### 3️⃣  Run the notebook

jupyter main

Then open `main.ipynb` and follow the cells step by step.

## 🛠 Main Features

- **Fit and compare AFT models:**
    - Log-Logistic
    - Log-Normal
    - Weibull
- **Plot survival functions** for a selected sample.
- **Calculate CLV per customer.**
- **Analyze CLV** across customer segments (e.g., customer category, marital status).
- **Save plots and results** for reporting.

## 📊 Output Example

- Survival curves plot (`img.png`)
- CLV summary table by customer segment
- Model comparison based on AIC

## ✅ Requirements

The main libraries used are:

- `lifelines`
- `matplotlib`
- `pandas`

(See `requirements.txt` for the full list.)

## ✍️ Report

A sample LaTeX report is included to help you format your findings for submission or presentation.

## 🤝 Contributions

Feel free to fork this repo and open pull requests to improve the analysis or add new features.

## License
MIT License.
---

✅ **What’s covered:**
- Setup instructions (clone, install, run),
- Repo structure overview,
- Description of the main functionality,
- Notes about outputs and dependencies.

Do you want to **link the actual Overleaf report** or customize the repo name/URL? 😊


