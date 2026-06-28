# 🏦 Loan Default Prediction Using Machine Learning

## 📌 Overview

The **Loan Default Prediction System** is a Machine Learning project that predicts whether a loan application is likely to be approved or rejected based on applicant information. The project uses historical loan data to train a **Random Forest Classifier** and provides predictions through an interactive **Streamlit** web application.

---

## 🎯 Objectives

* Predict loan approval status using Machine Learning.
* Reduce manual decision-making in the loan approval process.
* Build an interactive and user-friendly web application.
* Demonstrate practical use of Machine Learning in the banking domain.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Streamlit
* Joblib

---

## 🤖 Machine Learning Algorithm

* Random Forest Classifier

---

## 📂 Project Structure

```text
Loan_Default_Prediction/
│
├── dataset/
│   └── loan_data.csv
│
├── model/
│   └── loan_model.pkl
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
└── images/
```

---

## 📊 Dataset Features

* Gender
* Married
* Dependents
* Education
* Self Employed
* Applicant Income
* Coapplicant Income
* Loan Amount
* Loan Amount Term
* Credit History
* Property Area

**Target Variable**

* Loan Status (Approved / Rejected)

---

## ⚙️ Project Workflow

1. Load the dataset.
2. Clean the data and handle missing values.
3. Encode categorical variables.
4. Perform exploratory data analysis (EDA).
5. Split the dataset into training and testing sets.
6. Train the Random Forest model.
7. Evaluate the model using accuracy, classification report, and confusion matrix.
8. Save the trained model.
9. Build a Streamlit web application for predictions.

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/sattipandu-29/Loan_Default_Prediction.git
```

Navigate to the project folder:

```bash
cd Loan_Default_Prediction
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

---

## 📈 Model Performance

* Algorithm: Random Forest Classifier
* Accuracy: **75.61%**

---

## ✨ Features

* Data Cleaning
* Missing Value Handling
* Label Encoding
* Exploratory Data Analysis (EDA)
* Loan Prediction
* Interactive Streamlit Interface
* Model Evaluation

---

## 📸 Application Screenshots

You can add screenshots of:

* Home Page
* Prediction Page
* Loan Approved Result
* Loan Rejected Result

Store the images inside the **images/** folder and reference them here.

---

## 🔮 Future Improvements

* Deploy the application online.
* Add multiple Machine Learning algorithms for comparison.
* Improve prediction accuracy through feature engineering.
* Connect the application to a database.
* Add user authentication and prediction history.

---

## 👩‍💻 Developer

**PANDU**

Major Project: Loan Default Prediction Using Machine Learning
