import streamlit as st
import pandas as pd
import warnings 
warnings.filterwarnings('ignore')
import joblib

# st.title('START UP PROJECT')
# st.subheader('Built By Gomycode Daintree')

st.markdown("<h1 style = 'color: #5F0F40; text-align: center; font-family:TaHoma'>STARTUP PROFIT PREDICTOR</h1>", unsafe_allow_html = True)
# st.markdown("<h1 style = 'color: #5F0F40; text-align: center; font-family:TaHoma'>STARTUP PROFIT PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #FFB0B0; text-align: center; font-family: cursive '>Built By Gomycode Data Science Daintree Cohort</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)

st.image('pngwing.com (2).png')

st.header('Project Background Information', divider = True)
st.write("The overarching objective of this ambitious project is to meticulously engineer a highly sophisticated predictive model meticulously designed to meticulously assess the intricacies of startup profitability. By harnessing the unparalleled power and precision of cutting-edge machine learning methodologies, our ultimate aim is to furnish stakeholders with an unparalleled depth of insights meticulously delving into the myriad factors intricately interwoven with a startup's financial success. Through the comprehensive analysis of extensive and multifaceted datasets, our mission is to equip decision-makers with a comprehensive understanding of the multifarious dynamics shaping the trajectory of burgeoning enterprises. Our unwavering commitment lies in empowering stakeholders with the indispensable tools and knowledge requisite for making meticulously informed decisions amidst the ever-evolving landscape of entrepreneurial endeavors.")


st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

data = pd.read_csv('startUp.csv')
st.dataframe(data)

st.sidebar.image('pngwing.com (1).png', caption = 'Welcome User')

st.sidebar.markdown("<br>", unsafe_allow_html= True)
st.sidebar.markdown("<br>", unsafe_allow_html= True)

# Declare user Input variables
st.sidebar.subheader('Input Variables', divider = True)
rd_spend = st.sidebar.number_input('Research And Development Expense')
admin = st.sidebar.number_input('Administrative Expense')
mkt = st.sidebar.number_input('Marketing Expense')

# display users input
input_var = pd.DataFrame()
input_var['R&D Spend'] = [rd_spend]
input_var['Administration'] = [admin]
input_var['Marketing Spend'] = [mkt]

st.markdown("<br>", unsafe_allow_html = True)
# display the users input variable
st.subheader('Users Input Variables', divider= True)
st.dataframe(input_var)

# Import the scalers
admin_scaler = joblib.load('Administration_Scaler.pkl')
mkt_scaler = joblib.load('Marketing Spend_Scaler.pkl')
rd_spend_scaler = joblib.load('R&D Spend_Scaler.pkl')


# transform the users input with the imported scalers
input_var['R&D Spend'] = rd_spend_scaler.transform(input_var[['R&D Spend']])
input_var['Administration'] = admin_scaler.transform(input_var[['Administration']])
input_var['Marketing Spend'] = mkt_scaler.transform(input_var[['Marketing Spend']])

# st.dataframe(input_var)

model = joblib.load('StartUps.pkl')
predicted = model.predict(input_var)

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html=True)

# creating prediction and interpretation tab
prediction, inter = st.tabs(["Model Prediction", "Model Interpretation"])
with prediction:
    if prediction.button('Predict Profit'):
        prediction.success(f'The predicted profit for your organisation is: {predicted[0].round(2)}')


with inter:
    intercept = model.intercept_
    coef = model.coef_
    inter.write(f'A percentage increase in Reseach and Development Expense makes Profit to increase by {coef[0].round(2)} naira')
    inter.write(f'A percentage increase in Administration Expense makes Profit to reduce by {coef[1].round(2)} naira')
    inter.write(f'A percentage increase in Marketting Expense makes Profit to increase by {coef[2].round(2)} naira')
    inter.write(f'The value of Profit when none of these expenses were made is {intercept.round(2)} naira')
