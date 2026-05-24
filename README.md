# 5G Throughput Prediction with Transfer Learning

This repository contains the project developed for the **Network Data Analysis Laboratory (2025-2026)** course at **Politecnico di Milano**. 

## Objective
The goal of this project is to predict user throughput in a dense 5G deployment given user terminal and radio unit features. The project is split into two main components:
1. **Regression Task (12 points):** Throughput regression on the ACC Arena dataset, comparing a **Neural Network** and a **Random Forest** model using features like BLER, PRB, SINR, user traffic values, and features from the $X$ closest users[cite: 89, 90, 92, 94].
2. **Advanced Task (3 points):** **Transfer Learning** from the ACC Arena scenario to the Salt&Tar scenario, comparing the performance with the same model trained on a limited Salt&Tar training set.

## 📊 Dataset
The project utilizes the *5G High Density Demand (HDD) Dataset in Liverpool City Region, UK*, which includes two different venues:
* **ACC Arena:** 12k users, sampled every 3 seconds.
* **Salt & Tar:** 3k users, sampled every second.

## 🛠️ Methodology & Steps
Following the laboratory requirements, the project includes:
1. **Data Visualization & Analysis:** Initial exploratory data analysis (EDA).
2. **Data Preprocessing:** Cleaning, scaling, and structuring samples.
3. **Model Optimization:** Hyperparameters tuning with cross-validation.
4. **Performance Evaluation:** Comparing models using metrics such as MSE, MAE, Accuracy, and training duration.

## 👥 Authors
* **Giacomo Buscaglia** - *Politecnico di Milano*
* **Zarlene de Mesa** - *Politecnico di Milano*
* **Davide Roccuzzo** - *Politecnico di Milano*
