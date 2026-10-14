# 🎗️ Breast Cancer Classification Dashboard using KNN

<p align="center">
  <strong>An Interactive Machine Learning Dashboard for Exploring Breast Cancer Classification</strong>
</p>

<p align="center">
  Built with Python, Scikit-learn, and Streamlit to explore how machine learning can classify breast cancer samples using numerical features.
</p>

---

## 🚀 Overview

What if you could explore a machine learning model through an interactive dashboard instead of just reading its predictions in a notebook?

This project brings **K-Nearest Neighbors (KNN) classification** to life through a Streamlit web application built on the **Wisconsin Diagnostic Breast Cancer dataset**.

The model's hyperparameters are optimized using `GridSearchCV`, and the trained model is saved for use in the dashboard.

🔍 **Explore the data.**  
⚙️ **Understand the model.**  
📊 **Visualize the distribution.**  
🧪 **Experiment with numerical features and observe predictions.**

Whether you're exploring machine learning for the first time or reviewing a classification workflow, this dashboard provides a hands-on way to understand how a KNN model works.

> ⚠️ **Educational Disclaimer:** This project is intended for educational and machine learning demonstration purposes only. It is not a medical diagnostic tool. Its predictions must not be used for medical decisions, diagnosis, or treatment.

## ✨ Features

| Feature | What you can explore |
|---|---|
| 🧠 KNN Classification | See how the K-Nearest Neighbors algorithm classifies samples |
| 🎛️ Interactive Predictions | Enter numerical feature values and generate model predictions |
| ⚙️ Hyperparameter Tuning | Inspect the best parameters selected through `GridSearchCV` |
| 📈 Validation Results | View the cross-validation score reported by the model-selection process |
| 🔬 Dataset Exploration | Explore the dataset's numerical features and class distribution |
| 📊 Data Visualization | Visualize class distribution and other supported data insights |
| 💾 Saved Model | Load the trained model using Joblib |
| 🌐 Interactive Web App | Explore the project through a Streamlit interface |

*Feature scaling is handled by a saved model pipeline if scaling was included during training.*

## 🧬 Dataset at a Glance

The project uses the Wisconsin Diagnostic Breast Cancer dataset available through Scikit-learn.

| Dataset detail | Description |
|---|---|
| 📦 Samples | 569 |
| 🔢 Numerical features | 30 |
| 🏷️ Classes | Benign and malignant |
| 🎯 Problem type | Binary classification |
| 📚 Data source | Scikit-learn dataset |

The dataset contains numerical measurements describing characteristics of cell nuclei derived from digitized fine-needle aspirate images.

These measurements are used as input features for the classification model.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| 🐍 Python | Core programming language |
| 🧮 NumPy | Numerical operations |
| 🐼 Pandas | Data manipulation and exploration |
| 🤖 Scikit-learn | Machine learning and model evaluation |
| 🔎 GridSearchCV | Hyperparameter selection through cross-validation |
| 🎈 Streamlit | Interactive dashboard development |
| 📊 Matplotlib & Seaborn | Data visualization |
| 📈 Plotly | Interactive charts, where implemented |
| 💾 Joblib | Saving and loading the trained model |

## 🗂️ Project Structure

```text
breast-cancer-dashboard/
│
├── 📓 breastcan.ipynb          # Model development and experimentation
├── 🌐 app.py                   # Streamlit dashboard
├── 💾 breast_cancer_knn.joblib # Saved trained model or pipeline
├── 📦 requirements.txt         # Python dependencies
├── 📖 README.md                # Project documentation
└── 🙈 .gitignore                # Files excluded from Git
```

## ⚡ Getting Started

Follow these steps to run the dashboard locally.

### 1️⃣ Clone the repository

Replace the placeholder with your actual GitHub repository URL.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd breast-cancer-dashboard
```

### 2️⃣ Create a virtual environment

```bash
python -m venv .venv
```

### 3️⃣ Activate the environment

**Windows — Command Prompt**

```bat
.venv\Scripts\activate.bat
```

### 4️⃣ Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5️⃣ Launch the dashboard 🚀

Make sure `breast_cancer_knn.joblib` exists in the project directory.

```bash
streamlit run app.py
```

Streamlit will display a local URL in your terminal. Open that URL in your browser to interact with the dashboard.

## 🧠 How It Works

The project follows a typical machine learning classification workflow:

```text
📥 Load the Dataset
        ↓
🔍 Explore and Prepare the Data
        ↓
⚙️ Tune KNN Hyperparameters
        ↓
📏 Apply Feature Scaling (if included)
        ↓
🏋️ Train and Evaluate the Model
        ↓
💾 Save the Trained Model
        ↓
🌐 Load the Model in Streamlit
        ↓
🎛️ Enter Features & Explore Predictions
```

### Why KNN?

K-Nearest Neighbors is a supervised machine learning algorithm that predicts a sample's class by examining the classes of nearby samples in feature space.

Its behavior depends on choices such as the number of neighbors, distance metric, and weighting strategy. `GridSearchCV` helps identify a promising parameter configuration using cross-validation.

Because KNN relies on distances between samples, **feature scaling can be especially important** when the input features have different numerical ranges.

## 🎯 Learning Outcomes

Through this project, I explored:

- Building a binary classification workflow using Scikit-learn.
- Understanding the K-Nearest Neighbors algorithm.
- Tuning hyperparameters with `GridSearchCV`.
- Working with numerical datasets and feature distributions.
- Saving and loading trained models using Joblib.
- Integrating a machine learning model into a Streamlit application.
- Building an interactive interface for model predictions and exploration.

## 🔮 Future Improvements

Potential enhancements for the project include:

- 📊 Add a confusion matrix and classification report.
- 📈 Compare KNN with Logistic Regression, Decision Trees, and Random Forest.
- 🎯 Display precision, recall, F1-score, and ROC-AUC where appropriate.
- 🔍 Add feature-distribution and correlation visualizations.
- 🧪 Include model limitations and validation methodology in the dashboard.
- 🎨 Improve the interface with additional interactive charts and a more polished layout.

## 👨‍💻 Project Purpose

This project is a hands-on exploration of the machine learning workflow, from dataset exploration and model tuning to deployment through an interactive web application.

**Built to learn, experiment, and make machine learning more accessible.** 🚀
