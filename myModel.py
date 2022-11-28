import numpy as np
import cv2 as cv
from PIL import Image 
from IPython.display import Image as show_image  


from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.preprocessing import image
from keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions

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
print(arr)
