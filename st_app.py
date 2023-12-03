from transformers import pipeline
import streamlit as st
import joblib
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import numpy as np
import plotly.express as px
import random
from streamlit.components.v1 import iframe

# Download NLTK resources
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #89cff0; border-radius: 0.25rem; padding: 1rem; background-color: #89cff0;">{}</div>"""

# Model paths
MODEL_PATH = 'model/passmodel.pkl'
TOKENIZER_PATH ='model/tfidfvectorizer.pkl'
DATA_PATH ='data/drugsComTrain_raw.csv'

# Load vectorizer and model
vectorizer = joblib.load(TOKENIZER_PATH)
model = joblib.load(MODEL_PATH)

# Load stopwords and lemmatizer
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Function to clean text
def cleanText(raw_review):
    # 1. Delete HTML
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. Lowercase letters
    words = letters_only.lower().split()
    # 5. Remove stopwords
    meaningful_words = [w for w in words if not w in stop]
    # 6. Lemmatization
    lemmitize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    # 7. Join words with spaces
    return ' '.join(lemmitize_words)

# Function to extract top drugs
def top_drugs_extractor(condition, df):
    df_top = df[(df['rating'] >= 9) & (df['usefulCount'] >= 100)].sort_values(by=['rating', 'usefulCount'], ascending=[False, False])
    drug_lst = df_top[df_top['condition'] == condition]['drugName'].head(3).tolist()
    return drug_lst

# Load a pre-trained language model (e.g., GPT-2)
generator = pipeline("text-generation", model="gpt2")

# Feedback DataFrame
FEEDBACK_FILE = '/Users/tarun/MSAI/CS 5170/HealthPlanRecommendationSystem-main/feedback_data.csv'
feedback_df = pd.DataFrame(columns=['Predicted Condition', 'Top Drugs', 'Rating'])

def anonymize_feedback(feedback_df):
    # Anonymize by removing personally identifiable information
    anonymized_feedback_df = feedback_df.drop(columns=['User_ID'])  # Assuming 'User_ID' is a PII column
    return anonymized_feedback_df

def save_feedback_to_file(feedback_df):
    # Anonymize feedback before saving
    anonymized_feedback_df = anonymize_feedback(feedback_df)
    anonymized_feedback_df.to_csv(FEEDBACK_FILE, index=False)

def generate_health_plan(medical_condition, medicine_names):
    prompt = f"Patient has {medical_condition} and is prescribed {', '.join(medicine_names)}. The health plan is as follows:"

    # Generate health plan text
    health_plan = generator(prompt, max_length=200, num_return_sequences=1, temperature=0.7)[0]['generated_text']

    return health_plan

# Function to generate more information about the condition
def generate_condition_info(medical_condition):
    prompt = f"Tell me more about {medical_condition}."
    info = generator(prompt, max_length=200, num_return_sequences=1, temperature=0.7)[0]['generated_text']
    return info

# Streamlit app
def main():
    global feedback_df  # Use a global variable for feedback_df

    # Centered title with image
    col1, col2, col3 = st.columns(3)

    with col2:
        st.image("title.jpg")

    # st.image("title.jpg", width=200)
    # st.title("MEDICAL ADVISOR")
    st.markdown("<h1 style='text-align: center; color: #5cbdea;'>MEDICAL ADVISOR</h1>", unsafe_allow_html=True)

    # Set background color and style
    st.markdown(
        """
        <style>
        body {
            background-color: #89cff0;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    raw_text = st.text_area("Enter symptoms/test reports:", "")

    if st.button("Predict"):
        if raw_text != "":
            clean_text = cleanText(raw_text)
            clean_lst = [clean_text]

            tfidf_vect = vectorizer.transform(clean_lst)
            prediction = model.predict(tfidf_vect)
            predicted_cond = prediction[0]

            df = pd.read_csv(DATA_PATH)
            top_drugs = top_drugs_extractor(predicted_cond, df)

            st.write("Predicted Condition:", predicted_cond)  
            if st.button("Know More"):
                if predicted_cond:
                    # Generate more information about the condition
                    condition_info = generate_condition_info(predicted_cond[0])
                    st.subheader(f"More information about {predicted_cond[0]}:")
                    st.markdown(HTML_WRAPPER.format(condition_info), unsafe_allow_html=True)
         
            # st.write("Top Drugs for this Condition:", top_drugs)
            # Count the frequency of each drug
            drug_probabilities = [random.uniform(0, 1) for _ in range(len(top_drugs))]
            # Normalize probabilities to ensure they sum to 1
            drug_probabilities = np.array(drug_probabilities)
            drug_probabilities /= drug_probabilities.sum()

            # Generate random data for demonstration purposes
            drugs_count = pd.Series(np.random.choice(top_drugs, p=drug_probabilities, size=1000)).value_counts()

            # Display the pie chart for top drugs
            st.subheader("Top Drugs for this Condition:")
            st.plotly_chart(px.pie(values=drugs_count.values, names=drugs_count.index))
            # Generate health plan
            health_plan = generate_health_plan(predicted_cond, top_drugs)
            st.subheader("Health Plan")
            # Display the generated health plan with background color
            st.markdown(HTML_WRAPPER.format(health_plan), unsafe_allow_html=True)

            # Form setup
            form = st.form(key='feedback_form')

            # Feedback section
            form.subheader("Feedback")
            form.write("Rate the health plan:")

            # Rating slider
            rating = form.slider("Feedback Rating", 1, 10, 5)

            # Submit button
            submit_button = form.form_submit_button("Submit Feedback")

            form.write("")  # Add some space

            if submit_button:
                # Save feedback to DataFrame
                new_feedback = pd.DataFrame([{'Predicted Condition': predicted_cond[0], 'Top Drugs': ', '.join(top_drugs), 'Rating': rating}])
                feedback_df = pd.concat([feedback_df, new_feedback], ignore_index=True)
                print("Before saving feedback to CSV")
                save_feedback_to_file(feedback_df)
                print("After saving feedback to CSV")
                st.success("Thank you for your feedback! It has been saved.")

if __name__ == "__main__":
    main()
