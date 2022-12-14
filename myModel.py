import numpy as np
import cv2 as cv
from PIL import Image 
from IPython.display import Image as show_image 
import pandas as pd 
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import plotly.express as px 
import matplotlib.pyplot as plt
import numpy as np
print(sns.__version__)
print(cv.__version__)
print(Image.__version__)
print(pd.__version__)
print(tf.__version__)
print(np.__version__)
""" from tensorflow import keras 
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
 """

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions

IMAGE_PATH = "scar.jpg"
model = InceptionResNetV2(weights='imagenet', classes=1000)

def predic(img_path):
    img = Image.open(img_path).resize((299,299)) # model requirentment resulotion 299x299
    img = np.array(img)
    img = img.reshape(-1,299,299,3) # add one dim for model
    img = preprocess_input(img) # scale image -1 between 1 
    
    preds = model.predict(img)
    arr = decode_predictions(preds, top=5)[0] # model prediction array 
    
    return arr


arr = predic(img_path=IMAGE_PATH)

df = pd.DataFrame(arr, columns=['ID', 'Name', 'Prediction'])


print(df)