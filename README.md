# 🛍️ Mall Customer Segmentation & K-Means Clustering

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=priyanka19102005/Mall_Segmantation_Dataset&branch=main&mainModule=app.py)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

An interactive machine learning web application built with Streamlit and Scikit-Learn that clusters mall shoppers based on Annual Income, Spending Score, and Age using K-Means clustering.

---

## 📌 Features

- **Interactive 2D & 3D Clustering**: Real-time cluster computation with customizable $K$ (2 to 10) and dynamic centroid markers.
- **Elbow Method Curve**: Automatic calculation of WCSS (Inertia) to validate optimal cluster sizing.
- **Marketing Personas**: Automatic identification of high-value shoppers, budget spenders, impulsive spenders, and careful shoppers.
- **Real-Time Customer Classifier**: Enter prospective client metrics to predict their segment and recommended campaign strategy.
- **Data Export**: Export clustered customer data as CSV.

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/priyanka19102005/Mall_Segmantation_Dataset.git
cd Mall_Segmantation_Dataset
pip install -r requirements.txt
streamlit run app.py
```
