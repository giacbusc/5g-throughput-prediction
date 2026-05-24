# 5G Throughput Prediction with Transfer Learning

[cite_start]This repository contains the project developed for the **Network Data Analysis Laboratory (2025-2026)** course at **Politecnico di Milano**[cite: 4, 6]. 

## Objective
[cite_start]The goal of this project is to predict user throughput in a dense 5G deployment given user terminal and radio unit features[cite: 58, 88]. The project is split into two main components:
1. [cite_start]**Regression Task (12 points):** Throughput regression on the ACC Arena dataset, comparing a **Neural Network** and a **Random Forest** model using features like BLER, PRB, SINR, user traffic values, and features from the $X$ closest users[cite: 89, 90, 92, 94].
2. [cite_start]**Advanced Task (3 points):** **Transfer Learning** from the ACC Arena scenario to the Salt&Tar scenario, comparing the performance with the same model trained on a limited Salt&Tar training set[cite: 100, 101].

## 📊 Dataset
[cite_start]The project utilizes the *5G High Density Demand (HDD) Dataset in Liverpool City Region, UK*, which includes two different venues[cite: 60]:
* [cite_start]**ACC Arena:** 12k users, sampled every 3 seconds[cite: 61, 62].
* [cite_start]**Salt & Tar:** 3k users, sampled every second[cite: 61, 62].

## 🛠️ Methodology & Steps
[cite_start]Following the laboratory requirements, the project includes[cite: 8]:
1. [cite_start]**Data Visualization & Analysis:** Initial exploratory data analysis (EDA)[cite: 9].
2. [cite_start]**Data Preprocessing:** Cleaning, scaling, and structuring samples[cite: 11].
3. [cite_start]**Model Optimization:** Hyperparameters tuning with cross-validation[cite: 12, 13].
4. [cite_start]**Performance Evaluation:** Comparing models using metrics such as MSE, MAE, Accuracy, and training duration[cite: 14, 23].

## 👥 Authors
* **Giacomo Buscaglia** - *Politecnico di Milano*
* **Zarlene de Mesa** - *Politecnico di Milano*
* **Davide Roccuzzo** - *Politecnico di Milano*
