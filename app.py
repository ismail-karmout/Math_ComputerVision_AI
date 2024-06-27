import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
import os
from PIL import Image 
import streamlit as st 

genai.configure(api_key="")
model = genai.GenerativeModel('gemini-1.5-flash')
 
st.set_page_config(layout="wide")
 
st.title("Math & C.V & AI")
 
col1, col2 = st.columns([2, 1])

with col1:
    FRAME_WINDOW = st.image([])   

with col2:
    st.title("Result...")
    output_text_area = st.subheader("")   

 
st.markdown("""
    <style>
    .css-1lcbmhc {
        border-right: 3px solid #000;
        height: 100%;
        position: absolute;
        font-weight: bold;
        left: 50%;
        top: 0;
        color : orange
    }
    </style>
""", unsafe_allow_html=True)

 
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=True)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None: 
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (255, 0, 0), 10)
    elif fingers == [0, 1, 1, 1, 1]:
        canvas = np.zeros_like(img)
    return current_pos, canvas

def sendToAI(model, canvas, fingers):
    if fingers == [1, 0, 0, 0, 0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this Math problem", pil_image])
        # response = model.generate_content(["Guess the drawing : ", pil_image])
        
        return response.text
    return ""

prev_pos = None
canvas = None
image_combines = None
outtext = ""

# AIzaSyCB4fXGZEOBXGGoFVRu1MQnBNsaNZUfm1o
while cap.isOpened():
    success, img = cap.read()
    if not success or img is None:
        continue  

    img = cv2.flip(img, 1)
    
    if canvas is None:
        canvas = np.zeros_like(img)
        image_combines = img.copy()

    info = getHandInfo(img)
    
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        outtext = sendToAI(model, canvas, fingers)
        
    image_combines = cv2.addWeighted(img, 0.5, canvas, 0.7, 0)
    if outtext:
        
        output_text_area.markdown(f"<span style='color: orange;'>{outtext}</span>", unsafe_allow_html=True)
    
    FRAME_WINDOW.image(image_combines, channels="BGR")
    cv2.waitKey(1)
