import pandas as pd
import joblib
import gradio as gr

# Load the trained model and the exact column structure used during training
gb = joblib.load("gb_model.pkl")
model_columns = joblib.load("model_columns.pkl")


def predict_diabetes(age, bmi, systolic_bp, diastolic_bp, cholesterol_total,
                      hdl_cholesterol, ldl_cholesterol, triglycerides,
                      physical_activity, sleep_hours, alcohol, screen_time,
                      diet_score, waist_to_hip_ratio, heart_rate,
                      family_history, hypertension, cardiovascular,
                      gender, ethnicity, education, income, smoking, employment):

    # Build a raw dataframe matching original df columns
    raw = pd.DataFrame([{
        'age': age, 'bmi': bmi, 'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp, 'cholesterol_total': cholesterol_total,
        'hdl_cholesterol': hdl_cholesterol, 'ldl_cholesterol': ldl_cholesterol,
        'triglycerides': triglycerides,
        'physical_activity_minutes_per_week': physical_activity,
        'sleep_hours_per_day': sleep_hours,
        'alcohol_consumption_per_week': alcohol,
        'screen_time_hours_per_day': screen_time,
        'diet_score': diet_score, 'waist_to_hip_ratio': waist_to_hip_ratio,
        'heart_rate': heart_rate,
        'family_history_diabetes': family_history,
        'hypertension_history': hypertension,
        'cardiovascular_history': cardiovascular,
        'gender': gender, 'ethnicity': ethnicity,
        'education_level': education, 'income_level': income,
        'smoking_status': smoking, 'employment_status': employment
    }])

    # Encode exactly like training data
    raw_encoded = pd.get_dummies(raw, drop_first=True)

    # Align columns to match training (fills any missing dummy columns with 0)
    raw_encoded = raw_encoded.reindex(columns=model_columns, fill_value=0)

    prediction = gb.predict(raw_encoded)[0]
    probability = gb.predict_proba(raw_encoded)[0][1]

    label = "Diabetic" if prediction == 1 else "Not Diabetic"
    return f"{label} (Probability: {probability:.2%})"


interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="BMI", info="Your weight in kg divided by height in meters squared (e.g. 22.5)"),
        gr.Number(label="Systolic BP", info="Upper number when you measure blood pressure (e.g. 120)"),
        gr.Number(label="Diastolic BP", info="Lower number when you measure blood pressure (e.g. 80)"),
        gr.Number(label="Total Cholesterol", info="From your last blood test report (e.g. 170)"),
        gr.Number(label="HDL Cholesterol", info="The good cholesterol from your blood test report (e.g. 55)"),
        gr.Number(label="LDL Cholesterol", info="The bad cholesterol from your blood test report (e.g. 100)"),
        gr.Number(label="Triglycerides", info="Fat level in blood from your blood test report (e.g. 90)"),
        gr.Number(label="Physical Activity (mins/week)"),
        gr.Number(label="Sleep Hours Per Day"),
        gr.Number(label="Alcohol Consumption Per Week"),
        gr.Number(label="Screen Time (hrs/day)"),
        gr.Number(label="Diet Score", info="How healthy do you eat? 1 = very unhealthy, 10 = very healthy"),
        gr.Number(label="Waist to Hip Ratio", info="Waist size divided by hip size (e.g. 0.80)"),
        gr.Number(label="Heart Rate"),
        gr.Number(label="Family History of Diabetes (0=No, 1=Yes)"),
        gr.Number(label="Hypertension History (0=No, 1=Yes)"),
        gr.Number(label="Cardiovascular History (0=No, 1=Yes)"),
        gr.Dropdown(choices=["Male", "Female"], label="Gender"),
        gr.Dropdown(choices=["Hispanic", "White", "Asian", "Black", "Other"], label="Ethnicity"),
        gr.Dropdown(choices=["Highschool", "Graduate", "Postgraduate", "No formal"], label="Education Level"),
        gr.Dropdown(choices=["Lower-Middle", "Upper-Middle", "Low", "Middle", "High"], label="Income Level"),
        gr.Dropdown(choices=["Never", "Former", "Current"], label="Smoking Status"),
        gr.Dropdown(choices=["Employed", "Retired", "Student", "Unemployed"], label="Employment Status"),
    ],
    outputs=gr.Text(label="Prediction"),
    title="Diabetes Prediction",
    description="Enter your details below to predict the likelihood of diabetes."
)

if __name__ == "__main__":
    interface.launch()
