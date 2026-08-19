# ==========================================
# Employee Attrition Prediction
# Interactive Prediction
# ==========================================

import pandas as pd
import joblib


# ------------------------------------------
# 1. Load trained model
# ------------------------------------------

model = joblib.load(
    "models/logistic_regression_model.pkl"
)


# ------------------------------------------
# 2. Load preprocessor
# ------------------------------------------

preprocessor = joblib.load(
    "models/preprocessor.pkl"
)


# ------------------------------------------
# 3. Get employee details from user
# ------------------------------------------

print("\n======================================")
print("EMPLOYEE ATTRITION PREDICTION")
print("======================================")

age = int(input("Enter Age: "))

business_travel = input(
    "Enter BusinessTravel "
    "(Travel_Rarely / Travel_Frequently / Non-Travel): "
)

daily_rate = int(
    input("Enter DailyRate: ")
)

department = input(
    "Enter Department "
    "(Sales / Research & Development / Human Resources): "
)

distance_from_home = int(
    input("Enter DistanceFromHome: ")
)

education = int(
    input("Enter Education (1-5): ")
)

education_field = input(
    "Enter EducationField: "
)

environment_satisfaction = int(
    input("Enter EnvironmentSatisfaction (1-4): ")
)

gender = input(
    "Enter Gender (Male / Female): "
)

hourly_rate = int(
    input("Enter HourlyRate: ")
)

job_involvement = int(
    input("Enter JobInvolvement (1-4): ")
)

job_level = int(
    input("Enter JobLevel (1-5): ")
)

job_role = input(
    "Enter JobRole: "
)

job_satisfaction = int(
    input("Enter JobSatisfaction (1-4): ")
)

marital_status = input(
    "Enter MaritalStatus "
    "(Single / Married / Divorced): "
)

monthly_income = int(
    input("Enter MonthlyIncome: ")
)

monthly_rate = int(
    input("Enter MonthlyRate: ")
)

num_companies_worked = int(
    input("Enter NumCompaniesWorked: ")
)

overtime = input(
    "Enter OverTime (Yes / No): "
)

percent_salary_hike = int(
    input("Enter PercentSalaryHike: ")
)

performance_rating = int(
    input("Enter PerformanceRating (1-4): ")
)

relationship_satisfaction = int(
    input("Enter RelationshipSatisfaction (1-4): ")
)

stock_option_level = int(
    input("Enter StockOptionLevel (0-3): ")
)

total_working_years = int(
    input("Enter TotalWorkingYears: ")
)

training_times_last_year = int(
    input("Enter TrainingTimesLastYear: ")
)

work_life_balance = int(
    input("Enter WorkLifeBalance (1-4): ")
)

years_at_company = int(
    input("Enter YearsAtCompany: ")
)

years_in_current_role = int(
    input("Enter YearsInCurrentRole: ")
)

years_since_last_promotion = int(
    input("Enter YearsSinceLastPromotion: ")
)

years_with_curr_manager = int(
    input("Enter YearsWithCurrManager: ")
)


# ------------------------------------------
# 4. Create DataFrame
# ------------------------------------------

employee = pd.DataFrame([{

    "Age": age,

    "BusinessTravel": business_travel,

    "DailyRate": daily_rate,

    "Department": department,

    "DistanceFromHome": distance_from_home,

    "Education": education,

    "EducationField": education_field,

    "EnvironmentSatisfaction":
        environment_satisfaction,

    "Gender": gender,

    "HourlyRate": hourly_rate,

    "JobInvolvement": job_involvement,

    "JobLevel": job_level,

    "JobRole": job_role,

    "JobSatisfaction": job_satisfaction,

    "MaritalStatus": marital_status,

    "MonthlyIncome": monthly_income,

    "MonthlyRate": monthly_rate,

    "NumCompaniesWorked":
        num_companies_worked,

    "OverTime": overtime,

    "PercentSalaryHike":
        percent_salary_hike,

    "PerformanceRating":
        performance_rating,

    "RelationshipSatisfaction":
        relationship_satisfaction,

    "StockOptionLevel":
        stock_option_level,

    "TotalWorkingYears":
        total_working_years,

    "TrainingTimesLastYear":
        training_times_last_year,

    "WorkLifeBalance":
        work_life_balance,

    "YearsAtCompany":
        years_at_company,

    "YearsInCurrentRole":
        years_in_current_role,

    "YearsSinceLastPromotion":
        years_since_last_promotion,

    "YearsWithCurrManager":
        years_with_curr_manager
}])


# ------------------------------------------
# 5. Apply preprocessing
# ------------------------------------------

employee_processed = (
    preprocessor.transform(employee)
)


# ------------------------------------------
# 6. Predict probability
# ------------------------------------------

probability = model.predict_proba(
    employee_processed
)

leave_probability = probability[0][1]


# ------------------------------------------
# 7. Apply tuned threshold
# ------------------------------------------

threshold = 0.40

if leave_probability >= threshold:
    prediction = 1
else:
    prediction = 0


# ------------------------------------------
# 8. Display result
# ------------------------------------------

print("\n======================================")
print("PREDICTION RESULT")
print("======================================")

print(
    f"Probability of leaving: "
    f"{leave_probability * 100:.2f}%"
)

print(
    f"Decision threshold: {threshold}"
)

if prediction == 1:

    print(
        "Prediction: Employee is likely to leave."
    )

else:

    print(
        "Prediction: Employee is likely to stay."
    )