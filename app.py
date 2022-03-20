from ctypes import alignment
from tkinter import CENTER
import  numpy as np
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pickle
from PIL import Image
from streamlit_lottie import st_lottie
import json
import requests


model = pickle.load(open('D:\Mnist\Classification','rb'))

st.markdown("<h1 style='text-align: center; color: black;'>Digit Classification</h1>", unsafe_allow_html=True)
#st.title("Digit Prediction")

#image = Image.open('D:\Mnist\mnist-examples.jpg')

#st.image(image,width=500)

def load_lottiefile(filepath: str):
    with open('D:\Mnist\code.json.json', "r") as f:
        return json.load(f)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
    

lottie_coding = load_lottiefile("D:\Mnist\code.json.json")  # replace link to local lottie file
lottie_hello = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_l3sfdi9x.json")

st_lottie(
    lottie_hello,
    speed=1,
    reverse=False,
    loop=True,
    quality="low", # medium ; high
   
    height=300,
    width=300,
    key=None,
)

 



 

canvas_result = st_canvas(
    fill_color = "#ffffff",
    stroke_width = 10,
    stroke_color = "#ffffff",
    background_color = "#000000",
    height = 250,width = 250,
    drawing_mode = 'freedraw',
    key = "canvas",
)


if canvas_result.image_data is not None:
    
    img = cv2.cvtColor(canvas_result.image_data, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img.astype('uint8'), (28,28))
    img = img.reshape(1, 784)


if st.button('Predict'):
    
    prediction = model.predict(img)
    st.balloons()
    st.write(prediction)