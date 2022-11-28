import numpy as np
import cv2 as cv
from PIL import Image 
from IPython.display import Image as show_image  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px 

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions

import streamlit as st
from PIL import Image
import streamlit as st
import myModel

st.set_page_config("TrasferLearn", ":tada:", "wide")
st.title("Trasfer Learning")


# bilgisayardan resim indirme ve gösterme    
def load_image(image_file):
	img = Image.open(image_file)
	return img

image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

try:
  st.image(load_image(image_file),width=250)
except AttributeError:
  print("please upload image")

if image_file is not None : 
    pArr = myModel.predic(image_file)
    df = pd.DataFrame(pArr, columns=['ID', 'Name', 'Prediction'])
    df.drop('ID', inplace=True, axis=1)
    st.dataframe(df)

    fig=px.bar(df,x='Prediction',y='Name', orientation='h')
    st.write(fig)

