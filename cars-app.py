# Updated version - Dec 31 2023

import streamlit as st
import pandas as pd
import pickle
from datetime import datetime


st.write("""
# Used Car Price Estimation App

This app predicts the price of a used car
""")

st.sidebar.header('Tell us about your car:')

def user_input_features():

    year = st.sidebar.selectbox('Year', 
            (2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021))

    make = st.sidebar.selectbox("Make", 
        ('Audi', 'BMW', 'Mercedes-Benz', 'Volkswagen'))

    
    if make == 'Audi':
        model = st.sidebar.selectbox('Model',
            ('A1', 'A3', 'A4','A5', 'A6', 'Q3', 'Q5'))

    elif make == 'BMW':
        model = st.sidebar.selectbox('Model',
            ('116', '118', '318','320', '520', '530', 'X1', 'X3'))

    elif make == 'Mercedes-Benz':
        model = st.sidebar.selectbox('Model',
            ('A 180', 'B 180', 'C 180', 'C 200', 'C 220', 'E 200', 'E 220', 'Vito'))

    else:
        # Note: There is no Jetta in the selection sidebar b/c there were very few Jettas in the dataset
        # All data about Jetta was removed when the dataset was restricted to makes and models with more
        # than 100 data points
        model = st.sidebar.selectbox('Model',
            ('Caddy','Golf', 'Passat', 'Polo', 'Tiguan', 'Touran'))

    

    fuel = st.sidebar.selectbox('Fuel',
            ('Gasoline', 'Diesel', 'Hybrid: Gas-electric', 'Electric'))

    # For simplicity, kept only the manual and automatic options
    gear = st.sidebar.selectbox('Gear',
            ('Manual', 'Automatic'))

    mileage = st.sidebar.slider('Mileage', 0, 500000, 20000)

    #hp = st.sidebar.slider('HP', 0,800,150)
    hp = 200


    

    data = {'mileage': mileage,
            'hp': hp,
            'year':year,
            'make': make,
            'model': model,
            'fuel': fuel,
            'gear': gear
            }
    
    # index = [0] or index=[1] just initializes the index that shows next to the line of data
    #features = pd.DataFrame(data, index=[0])
    features = pd.DataFrame(data, index=[1])

    return features

X_test = user_input_features()

st.subheader('Here is the information you entered:')
st.write(X_test.drop(columns=['hp']))




# Create a new column: 'age'
X_test['age'] = datetime.now().year - X_test['year']

# Drop the 'year' column and re-order the columns in the correct order that fits the training data pipeline
X_test = X_test.drop('year', axis=1)
X_test = X_test[['mileage','hp','age','make','model','fuel','gear']]

# These two lines are for debugging purposes
#st.subheader('Re-display Xtest:')
#st.write(X_test)


# Load the saved pickled model
load_model = pickle.load(open('cars_model.pkl','rb'))

X_test_prepared = load_model['final_pipeline'].transform(X_test)


y_test_prediction = load_model['final_model'].predict(X_test_prepared)


if y_test_prediction < 300:
    # Set the minimum price of a car to 300: car sold for scrap parts
    y_test_prediction = 300

st.subheader('Predicted price of your car:')
st.write(int(y_test_prediction))

