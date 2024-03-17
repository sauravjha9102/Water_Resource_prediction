import numpy as np
import pickle 
import streamlit as st

#loading the saved model 
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

def Extra_resource_pred(input_data):
    # Convert input data to numeric values
    input_data_numeric = [float(value) for value in input_data]

    input_data_as_numpy_array = np.asarray(input_data_numeric)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'Extra Water Resource is not Needed'
    else:
        return 'Extra Water Resource is Needed'
    

def main():
    # Giving title
    st.title('Extra water resource prediction')

    # Getting input from the user
    year = st.text_input("YEAR")
    Rainfall = st.text_input("Rainfall")
    ground_level = st.text_input("ground_level")
    Avg_hum = st.text_input("Average_Humidity")
    Avg_tem = st.text_input("Average_temp")

    # Code for Prediction
    pred = ''

    # Creating a button for prediction
    if st.button("prediction"):
        pred = Extra_resource_pred([year, Rainfall, ground_level, Avg_hum, Avg_tem])

    st.success(pred)

if __name__ == '__main__':
    main()
