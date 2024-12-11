from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import json
import re
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain.llms import HuggingFacePipeline
from functools import lru_cache

app = Flask(__name__)

# Load models from disk (assumed models are pre-trained and saved in 'models' directory)
model_paths = {
    'kmeans': 'models/kmeans_model.pkl',
    'scaler': 'models/scaler.pkl',
    'multi_target_classifier': 'models/multi_target_classifier_model.pkl',
    'rf_regressor_personal_loan': 'models/rf_regressor_personal_loan_model.pkl',
    'rf_regressor_home_loan': 'models/rf_regressor_home_loan_model.pkl',
    'rf_regressor_credit_card': 'models/rf_regressor_credit_card_model.pkl'
}

kmeans = joblib.load(model_paths['kmeans'])
scaler = joblib.load(model_paths['scaler'])
multi_target_classifier = joblib.load(model_paths['multi_target_classifier'])
rf_regressor_personal_loan = joblib.load(model_paths['rf_regressor_personal_loan'])
rf_regressor_home_loan = joblib.load(model_paths['rf_regressor_home_loan'])
rf_regressor_credit_card = joblib.load(model_paths['rf_regressor_credit_card'])

print("Models loaded successfully!")


@lru_cache(maxsize=None)
def load_model():
    model_name = "KingNish/Qwen2.5-0.5b-Test-ft"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model


def process_form_data(form_data):
    """Process the form data into a dictionary for model predictions."""
    customer_data = {
        'name': form_data.get('name', ''),
        'age': int(form_data.get('age', 0)),
        'gender': form_data.get('gender', ''),
        'marital_status': form_data.get('marital_status', ''),
        'education': form_data.get('education', ''),
        'occupation': form_data.get('occupation', ''),
        'salary': int(form_data.get('salary', 0)),
        'loan_amount': int(form_data.get('loan_amount', 0)),
        'credit_limit': int(form_data.get('credit_limit', 0)),
        'credit_utilization': int(form_data.get('credit_utilization', 0)),
        'emi_paid': int(form_data.get('emi_paid', 0)),
        'tenure_months': int(form_data.get('tenure_months', 0)),
        'max_dpd': int(form_data.get('max_dpd', 0)),
        'default_status': int(form_data.get('default_status', 0)),
        'account_balance': int(form_data.get('account_balance', 0)),
        'Credit Card': int(form_data.get('credit_card', 0)),
        'Home Loan': int(form_data.get('home_loan', 0)),
        'Personal Loan': int(form_data.get('personal_loan', 0)),
        'enquiry_amount': int(form_data.get('enquiry_amount', 0)),
        'unique_products_enquired': int(form_data.get('unique_products_enquired', 0)),
        'total_enquiries': int(form_data.get('total_enquiries', 0)),
        'transaction_amount': int(form_data.get('transaction_amount', 0)),
        'is_salary': int(form_data.get('is_salary', 0))
    }
    return customer_data


# Function to generate insights based on customer data
# Function to generate insights based on customer data
def generate_insights(customer_data):
    tokenizer, model = load_model()

    # Create a prompt from the customer data
    prompt = f"""
    Generate a personalized summarised insight about the following customer based on their data whether it can be a good customer to onboard as a banking entity:

    - Name: {customer_data['name']}
    - Age: {customer_data['age']}
    - Gender: {customer_data['gender']}
    - Marital Status: {customer_data['marital_status']}
    - Education: {customer_data['education']}
    - Occupation: {customer_data['occupation']}
    - Salary: ₹{customer_data['salary']:,.2f}
    - Loan Amount: ₹{customer_data['loan_amount']:,.2f}
    - Credit Limit: ₹{customer_data['credit_limit']:,.2f}
    - Credit Utilization: {customer_data['credit_utilization']:.4%}
    - EMI Paid: {customer_data['emi_paid']}
    - Tenure Months: {round(float(customer_data['tenure_months']), 2)}
    - Max DPD: {customer_data['max_dpd']}
    - Default Status: {int(customer_data['default_status'])}
    - Account Balance: ₹{customer_data['account_balance']:,.2f}
    - Credit Card: {customer_data['Credit Card']}
    - Home Loan: {customer_data['Home Loan']}
    - Personal Loan: {customer_data['Personal Loan']}
    - Enquiry Amount: ₹{customer_data['enquiry_amount']:,.2f}
    - Unique Products Enquired: {customer_data['unique_products_enquired']}
    - Total Enquiries: {customer_data['total_enquiries']}
    - Transaction Amount: ₹{customer_data['transaction_amount']:,.2f}
    - Is Salary Account: {'Yes' if customer_data['is_salary'] == 1 else 'No'},
    
    Here are the Summarised Insights about {customer_data['name']}:
    """
    # Initialize the query pipeline with increased max_length
    query_pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        max_length=6000,  # Increase max_length
        max_new_tokens=500,  # Control the number of new tokens generated
        device_map="auto",
    )

    llm = HuggingFacePipeline(pipeline=query_pipeline)

    insights = llm(prompt)
    return insights


def process_customer_data(json_data, scaler):
    # Convert JSON data to a DataFrame
    customer_data = pd.DataFrame.from_dict(json_data, orient='index').T

    # Drop columns not needed for clustering (based on your clustering model)
    clustering_data = customer_data[['age', 'salary', 'loan_amount', 'credit_limit', 'credit_utilization',
                                     'emi_paid', 'tenure_months', 'max_dpd', 'default_status',
                                     'enquiry_amount', 'unique_products_enquired', 'total_enquiries',
                                     'transaction_amount', 'account_balance', 'is_salary', 'Credit Card',
                                     'Home Loan', 'Personal Loan']]
    clustering_data = customer_data[[
        'age', 'salary', 'loan_amount', 'credit_limit', 'credit_utilization',
        'emi_paid', 'tenure_months', 'max_dpd', 'default_status',
        'enquiry_amount', 'unique_products_enquired', 'total_enquiries',
        'transaction_amount', 'account_balance', 'is_salary', 'Credit Card',
        'Home Loan', 'Personal Loan'
    ]]

    # Handle missing values by replacing with 0 (or any appropriate strategy)
    clustering_data.fillna(0, inplace=True)

    scaled_data = scaler.transform(clustering_data)

    return customer_data, scaled_data


def predict_customer_segment(scaled_data, kmeans):
    # Predict the customer segment using your pre-trained KMeans model
    customer_segment = kmeans.predict(scaled_data)
    return customer_segment[0]


def recommend_product_and_loan(json_data, kmeans, scaler, multi_target_classifier, rf_regressor_personal_loan,
                               rf_regressor_home_loan, rf_regressor_credit_card):
    # Convert JSON data to a DataFrame and scale the data
    customer_data, scaled_data = process_customer_data(json_data, scaler)

    # Step 1: Predict customer segment using KMeans
    customer_segment = predict_customer_segment(scaled_data, kmeans)

    # Add the predicted customer segment back to the customer_data DataFrame
    customer_data['customer_segment'] = customer_segment

    customer_data = customer_data[['age', 'salary', 'loan_amount', 'credit_limit', 'credit_utilization',
                                   'emi_paid', 'tenure_months', 'max_dpd', 'default_status',
                                   'enquiry_amount', 'unique_products_enquired', 'total_enquiries',
                                   'transaction_amount', 'account_balance', 'is_salary', 'Credit Card',
                                   'Home Loan', 'Personal Loan', 'customer_segment']]

    # Prepare for product recommendation using Random Forest Classifier
    X_classification_prod = customer_data.drop(columns=['Credit Card', 'Home Loan', 'Personal Loan'])
    X_classification_amt = customer_data
    print(X_classification_prod.columns)

    # Step 2: Predict probabilities for each product using the multi-output classifier
    prob_credit_card = [estimator.predict_proba(X_classification_prod)[:, 1] for estimator in
                        multi_target_classifier.estimators_]

    # Combine probabilities into a Series
    product_probabilities = pd.Series({
        'Credit Card': prob_credit_card[0][0],  # Since it's for one customer, we get the first value
        'Home Loan': prob_credit_card[1][0],
        'Personal Loan': prob_credit_card[2][0]
    })

    print(product_probabilities)

    # Identify the most probable product
    recommended_product = product_probabilities.idxmax()
    recommended_probability = product_probabilities.max()

    recommendation = f"Recommended Product: {recommended_product} (Probability: {recommended_probability:.2f})"

    # Step 3: Predict loan amounts or credit limits based on the recommended product
    if recommended_product == 'Personal Loan':
        predicted_loan_amount_personal = rf_regressor_personal_loan.predict(
            X_classification_amt.drop(columns=['loan_amount']))
        recommendation += f"\nPredicted Loan Amount: {predicted_loan_amount_personal[0]:,.2f}"

    elif recommended_product == 'Home Loan':
        predicted_loan_amount_home = rf_regressor_home_loan.predict(X_classification_amt.drop(columns=['loan_amount']))
        recommendation += f"\nPredicted Loan Amount: {predicted_loan_amount_home[0]:,.2f}"

    elif recommended_product == 'Credit Card':
        predicted_credit_limit = rf_regressor_credit_card.predict(X_classification_amt.drop(columns=['loan_amount']))
        recommendation += f"\nPredicted Credit Limit: {predicted_credit_limit[0]:,.2f}"

    if recommended_probability < 0.5:
        recommendation = "No suitable product recommendations found for this customer."

    return recommendation, customer_segment, product_probabilities


def clean_and_extract_insight(insights):
    # Remove unwanted characters (non-alphanumeric characters except spaces)
    cleaned_insight = re.sub(r'[^a-zA-Z0-9\s]', '', insights)

    # Extract the portion after "Summarised Insight"
    if "Here are the Summarised Insights about" in cleaned_insight:
        extracted_insight = cleaned_insight.split("Here are the Summarised Insights about")[1].strip().split("\n\n")[0]
    else:
        extracted_insight = cleaned_insight.strip()

    return extracted_insight


@app.route('/', methods=['GET', 'POST'])
def index():
    recommendation = None
    insights = None
    customer_segment = None
    product_probabilities = None
    customer_data = {}

    if request.method == 'POST':
        customer_data = process_form_data(request.form)

        if 'generate_insights' in request.form:
            insights = generate_insights(customer_data)
            insights = [line.strip() for line in insights.split("\n") if line.strip()]

        elif 'generate_recommendation' in request.form:
            recommendation, customer_segment, product_probabilities = recommend_product_and_loan(
                customer_data, kmeans, scaler, multi_target_classifier,
                rf_regressor_personal_loan, rf_regressor_home_loan, rf_regressor_credit_card
            )

    return render_template('index.html', recommendation=recommendation, insights=insights,
                           customer_segment=customer_segment, product_probabilities=product_probabilities,
                           customer_data=customer_data)

if __name__ == '__main__':
    app.run(debug=True)