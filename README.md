# Employee Attrition Prediction using Logistic Regression

## 📌 Project Overview

This project is a **Machine Learning classification project** that predicts whether an employee is likely to leave an organization using **Logistic Regression**.

The project uses employee-related information such as job role, income, job satisfaction, overtime, stock options, experience, and other HR attributes to predict employee attrition.

The project covers the complete workflow from **data preprocessing to model prediction and Streamlit deployment**.

### Project Workflow

```text
Raw Employee Dataset
        ↓
Data Exploration
        ↓
Data Preprocessing
        ↓
Train-Test Split
        ↓
Feature Encoding & Scaling
        ↓
Logistic Regression
        ↓
Model Evaluation
        ↓
Threshold Tuning
        ↓
Save Model & Preprocessor
        ↓
Prediction
        ↓
Streamlit Application
```

---

## 🎯 Objective

The main objective of this project is to build a classification model that predicts whether an employee is likely to leave the organization.

### Target Variable

The target column is:

```text
Attrition
```

| Value | Meaning         |
| ----- | --------------- |
| `0`   | Employee stays  |
| `1`   | Employee leaves |

Since the target has two possible outcomes, this is a **binary classification problem**.

---

# 🧠 Algorithm Used

## Logistic Regression

Logistic Regression is a supervised machine learning algorithm commonly used for classification problems.

Unlike Linear Regression, which predicts continuous values, Logistic Regression predicts the **probability of a class**.

In this project:

```text
Employee Features
       ↓
Logistic Regression
       ↓
Probability of Attrition
       ↓
Classification Threshold
       ↓
Stay / Leave
```

The model produces an attrition probability, and a threshold is used to convert that probability into a final class prediction.

---

# 📊 Dataset

The project uses an employee dataset containing:

```text
1470 rows
35 columns
```

The dataset contains information about employees, including:

* Age
* Business Travel
* Daily Rate
* Department
* Distance From Home
* Education
* Education Field
* Environment Satisfaction
* Job Involvement
* Job Level
* Job Role
* Job Satisfaction
* Monthly Income
* Monthly Rate
* Num Companies Worked
* OverTime
* Percent Salary Hike
* Performance Rating
* Relationship Satisfaction
* Stock Option Level
* Total Working Years
* Training Times Last Year
* Work-Life Balance
* Years At Company
* Years In Current Role
* Years Since Last Promotion
* Years With Current Manager
* And other employee-related features

The target variable is:

```text
Attrition
```

---

# 📈 Attrition Distribution

Before training the model, the distribution of the target variable was analyzed.

![Attrition Distribution](images/attrition_distribution.png)

The dataset contains more employees who stayed than employees who left.

This creates a **class imbalance**, which is important to consider when evaluating the model.

Because of this imbalance, accuracy alone is not enough to judge model performance.

---

# 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**
* **Seaborn**
* **Joblib**
* **Streamlit**
* **Git**
* **GitHub**

---

# 📁 Project Structure

```text
Employee_Attrition_Logistic_Regression/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── images/
│   ├── attrition_distribution.png
│   ├── project_structure.png
│   ├── confusion_matrix_original.png
│   ├── roc_curve.png
│   ├── streamlit_input.png
│   └── streamlit_prediction.png
│
├── model/
│   ├── logistic_regression_model.pkl
│   └── preprocessor.pkl
│
├── main.py
├── predict.py
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

### Project Structure Screenshot

![Project Structure](images/project_structure.png)

---

# 🔄 Machine Learning Workflow

## 1. Data Loading

The dataset is loaded using Pandas.

```python
import pandas as pd

df = pd.read_csv(
    "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)
```

The dataset shape is:

```text
(1470, 35)
```

---

## 2. Data Exploration

The dataset was explored using:

* `head()`
* `shape`
* `info()`
* `describe()`
* Missing-value checking
* Duplicate checking
* Target distribution analysis

This step helps understand the structure and quality of the data before preprocessing.

---

# 3. Data Preprocessing

The dataset contains both **numerical** and **categorical** features.

### Numerical Features

Numerical features are processed using appropriate preprocessing techniques.

### Categorical Features

Categorical features are converted into numerical values using **One-Hot Encoding**.

A Scikit-learn preprocessing pipeline is used so that the same transformations can be applied during prediction.

---

# 4. Train-Test Split

The dataset was divided into training and testing datasets.

```text
Training Data:
1176 samples

Testing Data:
294 samples
```

The model is trained using the training data and evaluated on unseen testing data.

---

# 5. Feature Processing

After preprocessing and encoding:

```text
Training Shape:
(1176, 51)

Testing Shape:
(294, 51)
```

The original categorical features were transformed into numerical features, resulting in 51 processed features.

---

# 6. Model Training

Logistic Regression was used to train the classification model.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(...)

model.fit(X_train, Y_train)
```

The trained model learns the relationship between employee features and employee attrition.

---

# 📊 Model Evaluation

The model was evaluated using multiple metrics:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score
* ROC-AUC

Using multiple metrics is especially important because the dataset is imbalanced.

---

# 📌 Initial Model Performance

The initial Logistic Regression model achieved:

```text
Accuracy: 86.05%
```

The original confusion matrix was:

```text
[[237, 10],
 [ 31, 16]]
```

This means:

```text
True Negative  = 237
False Positive = 10
False Negative = 31
True Positive  = 16
```

---

# 🔲 Confusion Matrix

The confusion matrix shows how well the model classified employees into the two classes.

![Confusion Matrix](images/confusion_matrix_original.png)

### Interpretation

* **True Negative (TN):** Employee stayed and model predicted stay.
* **False Positive (FP):** Employee stayed but model predicted leave.
* **False Negative (FN):** Employee left but model predicted stay.
* **True Positive (TP):** Employee left and model predicted leave.

For employee attrition prediction, identifying employees who are likely to leave is particularly important.

---

# ⚖️ Class Imbalance and Threshold Tuning

The dataset contains significantly more employees who stayed than employees who left.

Therefore, the default classification threshold of `0.5` may not provide the best balance between precision and recall for the attrition class.

Different probability thresholds were tested.

The best threshold obtained was:

```text
Best Threshold: 0.4
Best F1-score: 0.5185
```

The final classification threshold was therefore set to:

```text
0.4
```

---

# 📈 Final Model Performance

After threshold tuning:

```text
Threshold: 0.4

Accuracy: 86.73%

ROC-AUC: 0.8115
```

Final confusion matrix:

```text
[[234, 13],
 [ 26, 21]]
```

### Final Results

| Metric                   | Result |
| ------------------------ | -----: |
| Accuracy                 | 86.73% |
| ROC-AUC                  | 0.8115 |
| Best F1-score            | 0.5185 |
| Classification Threshold |    0.4 |

---

# 📉 ROC Curve

The ROC curve is used to evaluate the model's ability to distinguish between employees who stay and employees who leave.

![ROC Curve](images/roc_curve.png)

The model achieved:

```text
ROC-AUC = 0.8115
```

An ROC-AUC of approximately `0.81` indicates that the model has good ability to distinguish between the two classes.

---

# 💾 Model Saving

The trained Logistic Regression model is saved using Joblib.

```text
model/logistic_regression_model.pkl
```

The preprocessing pipeline is also saved:

```text
model/preprocessor.pkl
```

Saving the preprocessing pipeline is important because new prediction data must go through the **same preprocessing steps** used during training.

---

# 🔮 Prediction

A separate prediction script is provided:

```text
predict.py
```

The prediction process is:

```text
New Employee Data
        ↓
Load Preprocessor
        ↓
Transform Input
        ↓
Load Saved Model
        ↓
Predict Probability
        ↓
Apply Threshold = 0.4
        ↓
Final Prediction
```

The model produces a prediction such as:

```text
Employee is likely to stay
```

or:

```text
Employee is likely to leave
```

---

# 🌐 Streamlit Application

A Streamlit application was created to provide an interactive interface for the trained machine learning model.

The application allows users to enter employee information and receive an attrition prediction.

---

## 🖥️ Streamlit Input

The user can enter employee-related information through the Streamlit interface.

![Streamlit Input](images/streamlit_input.png)

---

## 🔮 Streamlit Prediction

After entering the employee information, the application processes the input and generates the employee attrition prediction.

![Streamlit Prediction](images/streamlit_prediction.png)

---

# ▶️ How to Run the Project

## 1. Clone the Repository

```bash
git clone <your-github-repository-url>
```

## 2. Navigate to the Project

```bash
cd Employee_Attrition_Logistic_Regression
```

## 3. Create a Virtual Environment

```bash
python -m venv .venv
```

## 4. Activate the Virtual Environment

### Windows

```bash
.venv\Scripts\activate
```

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## 6. Train the Model

```bash
python main.py
```

## 7. Make a Prediction

```bash
python predict.py
```

## 8. Run the Streamlit Application

```bash
streamlit run app.py
```

---

# 📦 Requirements

All required Python libraries are listed in:

```text
requirements.txt
```

Install them using:

```bash
pip install -r requirements.txt
```

---

# 📄 File Description

| File / Folder      | Description                                     |
| ------------------ | ----------------------------------------------- |
| `data/`            | Contains the employee dataset                   |
| `images/`          | Contains project screenshots and visualizations |
| `model/`           | Contains the saved model and preprocessor       |
| `main.py`          | Data preprocessing, training and evaluation     |
| `predict.py`       | Makes predictions using the saved model         |
| `app.py`           | Streamlit application                           |
| `requirements.txt` | Project dependencies                            |
| `.gitignore`       | Files excluded from Git                         |
| `README.md`        | Project documentation                           |

---

# 📊 Project Results Summary

```text
Dataset:
1470 rows × 35 columns

Training Data:
1176 samples

Testing Data:
294 samples

Processed Training Features:
51

Initial Accuracy:
86.05%

Best Threshold:
0.4

Best F1-score:
0.5185

Final Accuracy:
86.73%

ROC-AUC:
0.8115
```

---

# 🧠 Key Concepts Learned

Through this project, I practiced:

* Binary Classification
* Logistic Regression
* Data Exploration
* Data Preprocessing
* Numerical Feature Processing
* Categorical Feature Encoding
* One-Hot Encoding
* Train-Test Split
* Class Imbalance
* Confusion Matrix
* Precision
* Recall
* F1-score
* ROC Curve
* ROC-AUC
* Probability Threshold Tuning
* Model Persistence using Joblib
* Prediction Pipeline
* Streamlit
* Machine Learning Project Structure

---

# 🚀 Future Improvements

Possible improvements for this project include:

* Hyperparameter tuning
* Feature selection
* Cross-validation
* Comparing Logistic Regression with other classification algorithms
* Improving recall for the attrition class
* Adding Precision-Recall curves
* Adding SHAP explainability
* Deploying the Streamlit application
* Monitoring model performance after deployment

---

# 👩‍💻 Author

**Susmitha Nallamolu**

AI / Machine Learning Engineer Learner

---

# ⭐ Conclusion

This project demonstrates an end-to-end **Employee Attrition Prediction system using Logistic Regression**.

It covers the complete process of:

```text
Data
 ↓
Exploration
 ↓
Preprocessing
 ↓
Model Training
 ↓
Evaluation
 ↓
Threshold Tuning
 ↓
Model Saving
 ↓
Prediction
 ↓
Streamlit Application
```

The final model achieved **86.73% accuracy** and an **ROC-AUC score of 0.8115** after threshold tuning.

The project provides practical experience in building, evaluating, saving, and deploying a Machine Learning classification model.
