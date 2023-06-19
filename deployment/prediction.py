
import streamlit as st
import pandas as pd
import numpy as np
import pickle 
import json
from PIL import Image

# load all files

with open('ab_model.pkl', 'rb') as file_1:
    ab_model = pickle.load(file_1)

# Pre-processing
with open('scale_feat.pkl', 'rb') as file_2:
    scale_feat = pickle.load(file_2)

with open('winsoriser.pkl', 'rb') as file_3:
    winsoriser = pickle.load(file_3)
    
# List Numeric & Category
with open('num_cols_sc.txt', 'r') as file_4:
    num_cols_sc = json.load(file_4)
    
with open('num_cols_nsc.txt', 'r') as file_5:
    num_cols_nsc = json.load(file_5)


def run():
    with st.form(key='from_diabetes'):
        
        st.title('Prediction Page')

        # sub header
        st.subheader('We calculate your metrics to calculate diabetes')

        # add pic
        image = Image.open('diabetes2.png')
        st.image(image)
        st.write('Columns below are parameter we would like to use to predict if a patient have a diabetes or not.')
        st.write('*`Please fill columns below to predict`*')

        gender = st.selectbox('Gender', [0,1], help='0 = Female, 1 = Male')

        age = st.number_input('Age', min_value=25, max_value=80,
                            value=45, step=1, help='Usia Pasien')
        
        hypertension = st.number_input('Hypertension', min_value=0, max_value=1 , value=0,
                                       step=1, help='have hypertension?')

        heart_disease = st.number_input('Heart Disease', min_value=0, max_value=1 , value=0,
                                       step=1, help='have heart disease?')

        bmi = st.number_input('Body Mass Index', min_value=5, max_value=80, 
                                value=30, step=5, help='Amount of BMI')
        
        HbA1c_level = st.number_input('Hemogloblin Level', min_value= 3, max_value= 10, 
                                      value= 6, help='Level of Hemogloblin 3-10')

        blood_glucose_level = st.slider('Glucose Level', 0, 400, 150, step=10, 
                                        help='Glucose amount in blood stream')
        

        st.markdown('---')
        submitted = st.form_submit_button('Predict')

        data_inf = {
                    'age': age,
                    'bmi': bmi,
                    'hemoglobin_level': HbA1c_level,
                    'blood_glucose_level': blood_glucose_level,
                    'gender': gender,
                    'hypertension': hypertension,
                    'heart_disease': heart_disease,
                }
        
        data_inf = pd.DataFrame([data_inf])
        st.dataframe(data_inf)

        if submitted:
            data_inf_sc = data_inf[num_cols_sc]
            data_inf_nsc = data_inf[num_cols_nsc]  

            # scalling
            data_inf_sc = scale_feat.transform(data_inf_sc)
            data_inf_sc = pd.DataFrame(data_inf_sc, columns=num_cols_sc)

            # Reset Index
            data_inf_sc.reset_index(drop= True, inplace= True)
            data_inf_nsc.reset_index(drop = True, inplace = True)
            data_final = pd.concat([data_inf_sc, data_inf_nsc], axis= 1)
            # modeling
            y_pred_inf = ab_model.predict(data_final)
                
            if y_pred_inf[0] == 1:
                st.write('# **`Prediction: You Have Diabetes`**')
            else:
                st.write('# **`Prediction: You do not Have Diabetes`**')

    if __name__ == '__main__':
        run()