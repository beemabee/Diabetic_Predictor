import streamlit as st
import eda
import prediction

navigation = st.sidebar.selectbox('pilih halaman: ', ('Explore', 'Prediction'))

if navigation == 'Explore':
    eda.run()
else:
    prediction.run()
