import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title='Diabetes Prediction', 
    layout='wide',
    initial_sidebar_state='expanded'
)

def run():
# title
    st.title('Diabetes Exploration')
    st.subheader('Explore The Diabetes Metrics & Dataset')
    # add pic
    image = Image.open('diabetes.png')
    st.image(image)
    st.markdown('---')
    
    markdown_text = '''
    ## Backgorund
    Firstly, diabetes is a prevalent and chronic health condition that affects a significant portion of the population worldwide.
    By providing a prediction model for diabetes, it can contribute to early detection and intervention, which is crucial in 
    managing the disease and preventing complications. Secondly, the integration of a diabetes prediction model in the web project 
    aims to enhance user experience and provide personalized health insights. Users can input their relevant health data, such as BMI,
    blood glucose levels, and other factors, to obtain a prediction of their likelihood of having diabetes.

    This information can empower individuals to make informed decisions about their health, seek appropriate medical attention
    if necessary, and adopt preventive measures to reduce the risk of diabetes. Overall, the inclusion of a diabetes prediction
    feature aligns with the objective of promoting health awareness and enabling users to take proactive steps towards their 
    well-being.

    ## Problem Statement 
    Using a dataset obtained from Kaggle, the goal is to build a predictive model that determines whether 
    individuals with specific characteristics are likely to have diabetes or not.

    ## Objective
    The objectives of this project are to preprocess the dataset, explore its features, analyze the data, 
    implement four different algorithms for predicting the target variable, and perform Hyperparameter Tuning 
    to optimize the models' performance.

    ## About Dataset
    |         Variable        |                                         Description                                           |
    |-------------------------|-----------------------------------------------------------------------------------------------|
    | Gender                  | Gender refers to the biological sex of the individual                                         |                        
    | Age                     | Age is an important factor as diabetes is more commonly diagnosed in older adults             |
    | hypertension            | Hypertension is a medical condition in which the blood pressure in the arteries is            |
    |                         | persistently elevated (1 = True, 0 = False)                                                   |
    | heart_disease           | Heart disease is another medical condition that is associated with an increased risk of       |
    |                         | developing diabetes                                                                           |
    | smoking_history         | Smoking history is also considered a risk factor for diabetes.                                |
    | bmi                     | BMI (Body Mass Index) is a measure of body fat based on weight and height                     |
    | HbA1c_level             | HbA1c (Hemoglobin A1c) level is a measure of a person's average blood sugar level over the    |
    |                         | past 2-3 months                                                                               |
    | blood_glucose_level     | Blood glucose level refers to the amount of glucose in the bloodstream at a given time        |
    | diabetes                | Diabetes is the target variable being predicted (1 = True, 0 = False)                         |
    
    '''

    st.markdown(markdown_text)
    st.markdown('---')


    st.subheader('Data Exploratory')
    st.markdown('---')

    st.write('### Patient Information')

    # show dataframe
    data = pd.read_csv('diabetes_prediction_dataset.csv')
    st.dataframe(data)
    st.markdown('---')

    # Distribusi Penderita Diabetes
    fig, ax = plt.subplots()
    plt.pie(data['diabetes'].value_counts(), 
            labels=['non-diabetic', 'diabetic'], 
            autopct='%1.1f%%',
            colors=['Grey', 'red'], 
            startangle=25,
            explode=[0.05, 0.05])
    plt.title('Diabetes Distribution')
    plt.axis('equal')
    st.pyplot(fig)
    '''
    Based on the chart above, around 91.5% of the total 100,000 patients do
    not suffer from diabetes and only **8.5%** of patients **do have diabetes**. 
    91.5% of total non-diabetic patients will be analyzed with health factors
    to predict whether the patient or others can get diabetes or not
    '''
    st.markdown('---')
    
    # visual barplot
    st.subheader('Chart Based on User Input ')
    st.markdown('---')

    choice = st.selectbox('Pick Numeric Columns: ', ('age', 
                                            'heart_disease',
                                            'bmi',
                                            'HbA1c_level', 'blood_glucose_level'))

    fig,ax = plt.subplots(figsize=(15,10))
    sns.kdeplot(data[choice], fill=True)
    ax.set_title(choice.capitalize()+' Ratio')
    st.pyplot(fig)
    st.markdown('---')

    # visual 2
     ## Categorical Data Plot
    pilihan_kategori = st.selectbox('Pick Category Column : ', ('gender','hypertension','smoking_history','diabetes'))
    fig= plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x=pilihan_kategori, hue='diabetes', palette='Set2')

    plt.xlabel(pilihan_kategori.capitalize())
    plt.ylabel('Count')
    plt.title(pilihan_kategori.capitalize()+' Ratio')
    plt.legend(title='Diabetes')

    st.pyplot(fig)

if __name__ == '__main__':
    run()