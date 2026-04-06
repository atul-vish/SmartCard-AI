# 🧠 SmartCard-AI — Customer Segmentation Dashboard

SmartCard-AI is an AI-powered customer segmentation web application that uses **K-Means Clustering** to analyze customer data and group users into meaningful segments such as High, Medium, and Low value customers. It provides an interactive dashboard where users can upload datasets, visualize clusters using **PCA**, and gain actionable business insights.

---

## Live Demo

[Click here to use the app](#)  
*(Add your Streamlit link here after deployment)*

---

## Features

- Upload your own dataset or use default data
- Automatic data preprocessing (handling missing values & encoding)
- Elbow Method for optimal cluster detection
- PCA-based visualization of customer segments
- Business insights dashboard
- Download segmented data

---

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- Matplotlib

---

## How to Run Locally

```bash
git clone https://github.com/your-username/SmartCard-AI.git
cd SmartCard-AI

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
streamlit run main.py
