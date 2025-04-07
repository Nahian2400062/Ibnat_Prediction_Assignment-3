# Ibnat_Prediction_Assignment-3

# 🚀 Predicting Fast-Growing Firms — Assignment 3

Author: **Nahian Ibnat**  
Course: Data Analysis 3 — MA in Economics (2025)  
Instructor: Gábor Békés  
Repo: [Ibnat_Prediction_Assignment-3](https://github.com/Nahian2400062/Ibnat_Prediction_Assignment-3)

---

## 📌 Project Overview

The goal of this assignment is to build a predictive model that identifies **fast-growing firms** using administrative panel data from 2010–2015. We define high-growth firms based on **log sales growth between 2012 and 2014**, and compare three predictive models:

- **Logistic Regression (Logit)**
- **Classification Tree (CART)**
- **Random Forest**

Threshold tuning and cost-sensitive classification are used to minimize business loss. We also analyze results across **two industry sectors**: **manufacturing** and **services**.

---

## 📂 Repository Contents

| File | Description |
|------|-------------|
| `fast_growth_prediction.R` | Final R script with full data prep, modeling, evaluation. Clean and commented. |
| `predicting-fast-growing-firms.html` | R Markdown output showing analysis, tables, graphs, and code. |
| `Ibnat_Summary Report.pdf` | Concise summary report (max 5 pages) targeted to data science leads. |
| `Assignment 3.pdf` | Full assignment description and grading rubric. |
| `cs_bisnode_panel.csv` | Panel data for 2010–2015 (firm-level). Not uploaded here due to size/sensitivity. |
| Graphs (`*.png`) | Expected Loss plots for each model and sector for report inclusion. |

---

## 🧠 Variable Dictionary

### General Firm Identifiers
- `comp_id`: Unique company identifier
- `year`: Reporting year
- `begin`, `end`: Accounting period start/end
- `founded_year`, `founded_date`: Date the firm was established
- `exit_year`, `exit_date`: If the firm exited, when

### Financial Variables
- `sales`: Annual sales (revenue)
- `curr_assets`, `fixed_assets`, `liq_assets`: Types of assets
- `personnel_exp`: Payroll and employee costs
- `profit_loss_year`: Net profit or loss
- `COGS`: Cost of goods sold
- `wages`, `inventories`, `material_exp`, etc.: Additional operating details

### Management / CEO Variables
- `ceo_count`: Number of top executives
- `foreign`: Foreign ownership (binary)
- `female`: Share of female top management
- `birth_year`: Birth year of CEO
- `inoffice_days`: CEO tenure
- `gender`: Gender mix
- `origin`: Firm origin (domestic or foreign)

### Sector & Geography
- `ind2`: 2-digit NACE industry code  
  - Manufacturing = 10–33  
  - Services = 55, 56, 95
- `region_m`: Region in Hungary (e.g. Central, West)
- `urban_m`: Urbanization measure

---

## ⚠️ Missing Values

- Missing values were handled **by filtering complete cases** for modeling.  
- Specifically:
  - Only firms with **non-missing sales** in 2012 and 2014 were used.
  - `drop_na()` was applied before model training.
- If more advanced imputation was needed, we recommend exploring:
  - Mean/mode imputation
  - kNN or MICE imputation (not used here)

---

## 📈 Model Performance Summary

| Model          | AUC   | Threshold | Expected Loss | Notes |
|----------------|-------|-----------|----------------|-------|
| Logistic       | 0.687 | 0.09      | $10,540        | High FP rate |
| CART           | 0.674 | 0.05      | $11,061        | No specificity |
| Random Forest  | 0.757 | 0.40      | **$7**         | Best performer ✅ |

---

## 🏭 Sector Comparison (Random Forest)

| Sector        | AUC   | Threshold Range | Expected Loss | Accuracy |
|---------------|--------|------------------|----------------|----------|
| Manufacturing | 0.692  | 0.34–0.56         | $0             | Perfect  |
| Services      | 0.784  | 0.39–0.55         | $0             | Perfect  |

---

## 💬 Key Takeaways

- **Random Forest** is the most reliable model under the business loss structure.
- Threshold tuning significantly reduced expected cost.
- **Industry-specific models** yield better performance than a pooled model.
- Ideal for use in targeting scale-ups for funding, support, or policy design.

---

## 📄 License

This repository is shared under the **MIT License**.


