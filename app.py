import streamlit as st
import pickle
import numpy as np

df=pickle.load(open('dataframe.pkl','rb'))
pipe=pickle.load(open('pipe_model.pkl','rb'))

st.title("Laptop Price Predictor App")
st.write("This spp work on the Machine Learning model, created from a small sample of real-worl data")
company=st.selectbox("Select the Manaufactor of Laptop",df['Company'].unique(),index=4)
typename=st.selectbox("Select the type of Laptop",df['TypeName'].unique())
cpu=st.selectbox("Processor Name",df['Cpu'].unique())
ram=st.selectbox("Amount of Ram on the system",[4,8,12,16,24,32,64,128],index=1)
gpu=st.selectbox("Graphics Card",df['Gpu'].unique())
os=st.selectbox("Operating Systems",df['OpSys'].unique())
os=st.radio("Operating Systems",df['OpSys'].unique(),index=2)
weight=st.slider("Weight of the laptop(in kg)",min_value=0.7,max_value=4.8,value=2.2,step=0.1)
touchscreen=st.selectbox("Does the laptop have a touch screen",['1','0'])
ips=st.selectbox("Does the laptop have an IPS display",['1','0'])
ppi=st.slider("Pixel Density on the Laptop",min_value=90,max_value=350,value=220,step=10)

if st.button("PREDICT PRICE"):
   query=np.array([[company,typename,cpu,ram,gpu,os,weight,touchscreen,ips,ppi]])
   op=pipe.predict(query)
   st.subheader("The laptop with above mentioned configuration will approximately cost around  ₹"+str(round(np.exp(op[0]))))
