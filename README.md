---
title: Customer Churn Intelligence & Retention System
emoji: 📊
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
pinned: false
---




# Customer Churn Intelligence & Retention System

## Overview
This project is an end-to-end machine learning system designed to predict customer churn and provide actionable insights to help businesses retain customers.

The system not only predicts churn probability but also explains the reasons behind churn and suggests personalized retention strategies.

---

## Problem Statement
Customers do not leave suddenly — their behavior changes gradually over time.  
These early warning signals are often missed, leading to customer loss and revenue impact.

Businesses lack a system to:
- Identify at-risk customers early
- Understand why customers churn
- Take timely retention actions

---

## Solution
To address this, I built a system that:

- Predicts churn probability for each customer
- Explains predictions using SHAP (Explainable AI)
- Segments customers into risk levels (High / Medium / Low)
- Recommends actions for customer retention
- Provides an interactive dashboard for decision-making

---

## Features

- 📊 Churn Prediction (Machine Learning Model)
- 🔍 Model Explainability (SHAP)
- 🎯 Risk Segmentation (High / Medium / Low)
- 💡 Action Recommendations
- 📈 Business Insights Dashboard
- 🔽 Downloadable Filtered Data
- 🎛 Interactive Filters (Risk, Actions, etc.)

---

## Dataset

The dataset includes:
- Demographics (Age, Gender, Country, City)
- Engagement metrics (Login Frequency, Session Duration)
- Purchase behavior (Total Purchases, Order Value)
- Customer interaction (Support Calls, Reviews)
- Financial data (Lifetime Value, Credit Balance)

Target Variable:
- `Churned` (0 = Active, 1 = Churned)

---

## Machine Learning Pipeline

1. Data Cleaning (duplicates, missing values)
2. Train-Test Split
3. Feature Transformation
4. Encoding & Scaling
5. Model Training
6. Model Evaluation
7. Save Model & Preprocessing Pipeline

---

## System Workflow

1. Upload dataset
2. Column validation
3. Data cleaning
4. Feature preprocessing
5. Churn probability prediction
6. SHAP-based explanation
7. Risk classification
8. Action recommendation
9. Dashboard visualization

---

## Dashboard Features

- Total Customers
- High / Medium / Low Risk Distribution
- Revenue at Risk
- Customer-level predictions
- Filters (Risk, Action)
- Download filtered dataset

---

## Key Concept

**Priority Score = Churn Probability × Lifetime Value**

This helps businesses focus on high-value customers who are at high risk.

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- SHAP (Explainable AI)
- Streamlit (for UI)
- Git & GitHub
