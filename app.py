
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pickle
import numpy as np


# Load the model
with open("breast.pkl", "rb") as f:
    model = pickle.load(f)

st.title("BREAST CANCER Prediction App")

ss=StandardScaler()
f1 = st.number_input("radius_mean",0,100,1,1)
f2 = st.number_input("texture_mean",0,100,1,1)
f3 = st.number_input("perimeter_mean",0,200,1,1)
f4 = st.number_input("area_mean",0,300,1,1)
f5 = st.number_input("smoothness_mean",0.0,1.0,0.01,0.01)
f6 = st.number_input("compactness_mean",0.0,1.0,0.01,0.01)
f7 = st.number_input("concavity_mean",0.0,1.0,0.01,0.01)
f8 = st.number_input("concave points_mean",0.0,1.0,0.01,0.01)
f9 = st.number_input("symmetry_mean",0.0,1.0,0.01,0.01)
f10 = st.number_input("fractal_dimension_mean",0.0,1.0,0.01,0.01)

f11 = st.number_input("radius_se",0,100,1,1)
f12 = st.number_input("texture_se",0,100,1,1)   
f13 = st.number_input("perimeter_se",0,200,1,1)   
f14 = st.number_input("area_se",0,300,1,1)
f15 = st.number_input("smoothness_se",0.0,1.0,0.01,0.01)
f16 = st.number_input("compactness_se",0.0,1.0,0.01,0.01)
f17 = st.number_input("concavity_se",0.0,1.0,0.01,0.01)
f18 = st.number_input("concave points_se",0.0,1.0,0.01,0.01)        
f19 = st.number_input("symmetry_se",0.0,1.0,0.01,0.01)
f20 = st.number_input("fractal_dimension_se",0.0,1.0 ,0.01,0.01)

f21 = st.number_input("radius_worst",0,100,1,1)
f22 = st.number_input("texture_worst",0,100,1,1)
f23 = st.number_input("perimeter_worst",0,200,1,1)
f24 = st.number_input("area_worst",0,300,1,1)
f25 = st.number_input("smoothness_worst",0.0,1.0,0.01,0.01)
f26 = st.number_input("compactness_worst",0.0,1.0,0.01,0.01)
f27  = st.number_input("concavity_worst",0.0,1.0,0.01,0.01)
f28 = st.number_input("concave points_worst",0.0,1.0,0.01,0.01)
f29 = st.number_input("symmetry_worst",0.0,1.0,0.01,0.01)
f30 = st.number_input("fractal_dimension_worst ",0.0,1.0,0.01,0.01)
features = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,
            f21,f22,f23,f24,f25,f26,f27 ,f28 ,f29 ,f30]
if st.button("Predict"):
    data = np.array([features])
    data = ss.fit_transform(data)
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("Prediction: Malignant ,cencerous")
    else:
        st.success("Prediction: Benign, not cencerous")
