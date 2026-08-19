# ==========================================
# Employee Attrition Prediction
# Streamlit Application
# ==========================================

import streamlit as st
import pandas as pd
import joblib


# ------------------------------------------
# 1. Load Model and Preprocessor
# ------------------------------------------

model = joblib.load(
    "models/logistic_regression_model.pkl"
)

preprocessor = joblib.load(
    "models/preprocessor.pkl"
)


# ------------------------------------------
# 2. Page Configuration
# ------------------------------------------

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide"
)


# ------------------------------------------
# 3. Title
# ------------------------------------------

st.title(
    "Employee Attrition Prediction"
)

st.write(
    "Predict whether an employee is likely "
    "to leave the organization using "
    "Logistic Regression."
)


# ------------------------------------------
# 4. Employee Information
# ------------------------------------------

st.header("Employee Information")


col1, col2, col3 = st.columns(3)


with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    business_travel = st.selectbox(
        "Business Travel",
        [
            "Travel_Rarely",
            "Travel_Frequently",
            "Non-Travel"
        ]
    )

    daily_rate = st.number_input(
        "Daily Rate",
        min_value=0,
        value=800
    )

    department = st.selectbox(
        "Department",
        [
            "Sales",
            "Research & Development",
            "Human Resources"
        ]
    )

    distance_from_home = st.number_input(
        "Distance From Home",
        min_value=0,
        value=10
    )

    education = st.number_input(
        "Education",
        min_value=1,
        max_value=5,
        value=3
    )

    education_field = st.selectbox(
        "Education Field",
        [
            "Life Sciences",
            "Medical",
            "Marketing",
            "Technical Degree",
            "Human Resources",
            "Other"
        ]
    )

    environment_satisfaction = st.number_input(
        "Environment Satisfaction",
        min_value=1,
        max_value=4,
        value=3
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female"
        ]
    )

    hourly_rate = st.number_input(
        "Hourly Rate",
        min_value=0,
        value=80
    )


with col2:

    job_involvement = st.number_input(
        "Job Involvement",
        min_value=1,
        max_value=4,
        value=3
    )

    job_level = st.number_input(
        "Job Level",
        min_value=1,
        max_value=5,
        value=1
    )

    job_role = st.selectbox(
        "Job Role",
        [
            "Sales Executive",
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

    job_satisfaction = st.number_input(
        "Job Satisfaction",
        min_value=1,
        max_value=4,
        value=3
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0,
        value=3000
    )

    monthly_rate = st.number_input(
        "Monthly Rate",
        min_value=0,
        value=15000
    )

    num_companies_worked = st.number_input(
        "Number of Companies Worked",
        min_value=0,
        value=2
    )

    overtime = st.selectbox(
        "OverTime",
        [
            "Yes",
            "No"
        ]
    )

    percent_salary_hike = st.number_input(
        "Percent Salary Hike",
        min_value=0,
        max_value=100,
        value=12
    )


with col3:

    performance_rating = st.number_input(
        "Performance Rating",
        min_value=1,
        max_value=4,
        value=3
    )

    relationship_satisfaction = st.number_input(
        "Relationship Satisfaction",
        min_value=1,
        max_value=4,
        value=3
    )

    stock_option_level = st.number_input(
        "Stock Option Level",
        min_value=0,
        max_value=3,
        value=0
    )

    total_working_years = st.number_input(
        "Total Working Years",
        min_value=0,
        value=5
    )

    training_times_last_year = st.number_input(
        "Training Times Last Year",
        min_value=0,
        value=3
    )

    work_life_balance = st.number_input(
        "Work Life Balance",
        min_value=1,
        max_value=4,
        value=3
    )

    years_at_company = st.number_input(
        "Years At Company",
        min_value=0,
        value=2
    )

    years_in_current_role = st.number_input(
        "Years In Current Role",
        min_value=0,
        value=1
    )

    years_since_last_promotion = st.number_input(
        "Years Since Last Promotion",
        min_value=0,
        value=0
    )

    years_with_curr_manager = st.number_input(
        "Years With Current Manager",
        min_value=0,
        value=1
    )


# ------------------------------------------
# 5. Prediction Button
# ------------------------------------------

if st.button(
    "Predict Attrition"
):

    # Create DataFrame

    employee = pd.DataFrame([{

        "Age": age,

        "BusinessTravel":
            business_travel,

        "DailyRate":
            daily_rate,

        "Department":
            department,

        "DistanceFromHome":
            distance_from_home,

        "Education":
            education,

        "EducationField":
            education_field,

        "EnvironmentSatisfaction":
            environment_satisfaction,

        "Gender":
            gender,

        "HourlyRate":
            hourly_rate,

        "JobInvolvement":
            job_involvement,

        "JobLevel":
            job_level,

        "JobRole":
            job_role,

        "JobSatisfaction":
            job_satisfaction,

        "MaritalStatus":
            marital_status,

        "MonthlyIncome":
            monthly_income,

        "MonthlyRate":
            monthly_rate,

        "NumCompaniesWorked":
            num_companies_worked,

        "OverTime":
            overtime,

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


    # --------------------------------------
    # 6. Preprocess
    # --------------------------------------

    employee_processed = (
        preprocessor.transform(employee)
    )


    # --------------------------------------
    # 7. Probability Prediction
    # --------------------------------------

    probability = model.predict_proba(
        employee_processed
    )

    leave_probability = probability[0][1]


    # --------------------------------------
    # 8. Apply Threshold
    # --------------------------------------

    threshold = 0.40

    if leave_probability >= threshold:

        prediction = 1

    else:

        prediction = 0


    # --------------------------------------
    # 9. Display Result
    # --------------------------------------

    st.header("Prediction Result")

    st.metric(
        "Probability of Leaving",
        f"{leave_probability * 100:.2f}%"
    )

    st.write(
        f"Decision Threshold: {threshold}"
    )


    if prediction == 1:

        st.error(
            "⚠️ Employee is likely to leave."
        )

    else:

        st.success(
            "✅ Employee is likely to stay."
        )