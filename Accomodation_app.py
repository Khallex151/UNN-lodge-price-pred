import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json



with open('Accomodation_price_columns.json', 'r') as f:
         data_columns = json.load(f)['data_columns']

with open('Accomodation_price_mae.txt', 'r') as f:
         mean_abs_err = f.read()

model = pickle.load(open('Accomodation_price_model.pickle', 'rb'))


def predict_price(model, Location, Accommodation_Type, Room_Size, Building_Age, 
                  Distance_From_School, Water_Source, Toilet, Gate, Toilet_Type, 
                  Location_Of_Water_Source, Electricity_Reliability):
    
    x = pd.DataFrame(columns=data_columns)
    # Create input array with zeros
    X = np.zeros(len(x.columns))

    # Assign numerical values to respective positions
    features = {
            "Location": np.where(x.columns == Location)[0][0],
            "Accomodation_Type": np.where(x.columns == Accommodation_Type)[0][0],
            "Room_Size": np.where(x.columns == Room_Size)[0][0],
            "Building_Age": np.where(x.columns == Building_Age)[0][0],
            "Distance_From_School": np.where(x.columns == Distance_From_School)[0][0],
            "Water_Source": np.where(x.columns == Water_Source)[0][0], 
            "Toilet": np.where(x.columns == Toilet)[0][0],
            "Gate": np.where(x.columns == Gate)[0][0],
            "Toilet_Type": np.where(x.columns == Gate)[0][0], 
            "Location_Of_Water_Source" : np.where(x.columns == Location_Of_Water_Source)[0][0], 
            "Electricity_Reliability" : np.where(x.columns == Electricity_Reliability)[0][0]
      }
    
    for feature, value in features.items():
        if value >= 0:
            X[value] = 1


    # Predict price using trained model
    return  model.predict([X])[0]



def user_input_features():
        
        price = predict_price(model, location, accomodation_type, room_size, building_age, 
                  dist_from_sch, water_source, toilet_loc, gate, toilet_type, 
                  location_of_water_source, elec_reliability)

        lower_range = round(price - float(mean_abs_err))
        higher_range = round(price + float(mean_abs_err))

        output = f'The Price for Your Lodge in {location} based on your description ranges between N{lower_range} - N{higher_range}💵'
        
        return output


###################### USER INTERFACE #####################

st.write("""
# UNN LODGES PRICE PREDICTION APP

This app predicts the **Prices of UNN Lodges** based on the choices of the user.

Data obtained from a  [survey](https://github.com/Francis-147/UNN-lodge-price-pred/blob/9d7491dd005d6f9b7f31f38ad834b68df0d82b51/Lodge%20Rent(Cleaned).csv) carried out in **UNN by 0'25 IT students of Education Innovation Unit**.
""")

st.subheader("Input Lodge Description")

location = st.selectbox('LOCATION',('Odenigwe', 'Behind Flat', 'Hilltop', 'Odim', 'Onuiyi', 'Vet Mountain','Greenhouse')).lower()
accomodation_type = st.selectbox('ACCOMODATION TYPE',('Single room', 'Self-con', 'Flat',)).lower()
room_size = st.selectbox('ROOM SIZE',('Small', 'Medium', 'Large', 'Very Large', 'Very Small')).lower()
building_age = st.selectbox('BUILDING AGE',('<5 years', '>20 years', '10-20 years')).lower()
dist_from_sch = st.selectbox('DISTANCE FROM SCHOOL',('<5 min', '15-30 min', '>30 min', '5-15 min')).lower()
water_source = st.selectbox('WATER SOURCE',('Public tap outside compound', 'Borehole outside compound', 'Private tap/tank in compound', 'In the unit', 'Government water','People selling water ', 'Well outside building', 'Tap inside the building ', 'No water..... ')).lower()
toilet_loc = st.selectbox('TOILET LOCATION',('En-suite', 'Shared', 'Outside')).lower()
gate = st.selectbox('GATE?',('No', 'Yes')).lower()
toilet_type = st.selectbox('TOILET TYPE',('Water Closet', 'Semi-Water Closet', 'Pit Latrine')).lower()
location_of_water_source = st.selectbox('LOCATION OF WATER SOURCE',('5-15 min', 'Within compound', '>30 min', '15-30 min')).lower()
elec_reliability = st.selectbox('ELECTRICITY RELIABILITY',('Moderately reliable', 'Very reliable', 'Unreliable', 'No Light', 'Very unreliable')).lower()


if st.button("Predict Price"):
    st.write(user_input_features())
      
