# streamlit run my_app.py
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os, urllib, cv2

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import time
import matplotlib.pyplot as plt
import math
import glob


# Streamlit encourages well-structured code, like starting execution in a main() function.
def RUN():

    
    #t.title('CASH RECOGNITION FOR INDIAN DATASET')
    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("Run,  Visualize")



    app_mode = st.sidebar.selectbox("Choose Mode",
        ["About","Run inference on single image","Show classification code","Performance Comparison"])
    if app_mode == "Show classification code":

        show_code()
    elif app_mode == "Run inference on single image":
        run_the_app()

    elif app_mode == 'About':
        overview()

    elif app_mode== 'Performance Comparison':
        run_openvino()


# This is the main app app itself, which appears when the user selects "Run the app".
def run_the_app():
    
    

    def load_image(image_path):
        
        #st.markdown("Loading Image...")
        test_image = cv2.imread(image_path)
        image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        return image


    option = ''   
    path = '/home/gason/demo_ltts/cash_recognition/test'

  
    files = [f for f in glob.glob(path + "**/*.jpg", recursive=True)]


    df = pd.DataFrame({ 'Image names': files})

    st.subheader("SELECT IMAGE")

    option = st.selectbox('',df['Image names'])
    submit = st.button("Submit")
    print(option)
    
    if submit:
       
        image = load_image(option)
            
        st.subheader("TEST IMAGE")

        st.image(image, use_column_width=True)
        st.write("**Image load successful.**")


        os.system("python /home/gason/demo_ltts/cash_recognition/test/classify.py --image %s"%option)
        print("option",option)


        latest_iteration = st.empty()
        bar = st.progress(0)

        for i in range(100):
            # Update the progress bar with each iteration.
            latest_iteration.text(f'Classification results {i+1}%')
            bar.progress(i+1)
            time.sleep(0.1)
    
        latest_iteration.text('Done!')
        
        output = load_image("/home/gason/demo_ltts/cash_recognition/demo_app/output.jpg")
        os.system("sudo rm /home/gason/demo_ltts/cash_recognition/demo_app/output.jpg")

        st.subheader("OUTPUT IMAGE")
        st.image(output, use_column_width=True)
        plot = load_image("/home/gason/demo_ltts/cash_recognition/demo_app/plot.png")
        st.subheader("RESULT")
        st.image(plot, use_column_width=True)

        st.write("**Image classification successful.**")
        st.balloons()
        del option
        del output

def show_code():
        train_code_path = '/home/gason/demo_ltts/cash_recognition/visualize/classify.txt'
        f = open(train_code_path)
        code = f.read()
        st.markdown(str(code))
       



def overview():

    pic_path = '/home/gason/demo_ltts/cash_recognition/demo_app/cash_pic.jpg'
    st.title("                    CASH RECOGNITION             ")
    st.image(pic_path, use_column_width=True)
    file_path = '/home/gason/demo_ltts/cash_recognition/About.txt'
    f = open(file_path)
    sentences = f.read()
    st.write(sentences)
    usage_path = '/home/gason/demo_ltts/cash_recognition/demo_app/usage_pic.png'
    st.image(usage_path, use_column_width=True)

def run_openvino():

    without_openvino = "/home/gason/demo_ltts/cash_recognition/demo_app/result_without_openvino.txt"
    with_openvino = "/home/gason/demo_ltts/cash_recognition/demo_app/result_with_openvino.txt"

    num1=''
    num2=''
    list1 = []
    with open(with_openvino) as f1:
        num1= f1.readline()
  

    list1.append(float(num1))

    with open(without_openvino) as f2:
        num2= f2.readline()


    

    list1.append(float(num2))
    fast = round(float(list1[1])/(float(list1[0])),3)
    

    height = list1[::-1]
    bars = ('Default classification on CPU','With OpenVINO IE on CPU')
    y_pos = np.arange(len(bars))
    plt.ylabel('Time taken (ms)')
    plt.xlabel('Environment')
    # Create bars
    plt.bar(y_pos, height,width=0.2)
     
    # Create names on the x-axis
    plt.xticks(y_pos, bars)
    plt.title("Object Classification Speed Comparison on CPU")
    
    res = '/home/gason/demo_ltts/cash_recognition/demo_app/results.png'

    # Show graphic
    plt.savefig(res)

    st.image(res, use_column_width=True)

    st.markdown("_Classification with OpenVINO Inference Engine on CPU is **%s** times faster than Default classification on CPU_"%fast)

    os.system("")









    


RUN()
