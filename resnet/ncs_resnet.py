from mvnc import mvncapi as mvnc
import sys
import numpy as np
import cv2
import time
from keras.applications.resnet50 import preprocess_input
from keras.preprocessing import image
from tqdm import tqdm
import os
import glob

def get_image(path):
    img=image.load_img(path, target_size=(224, 224))
    x=image.img_to_array(img)
    x=np.expand_dims(x, axis=0)
    x=preprocess_input(x)

    return np.asarray(x)

paths=glob.glob(os.path.join('updated_bad', 'Bad', '*.BMP'))

devices=mvnc.EnumerateDevices()

if len(devices)==0:
    print('No Devices Found')
    quit()

device = mvnc.Device(devices[0])
device.OpenDevice()

with open('reshape_graph.graph', mode='rb') as f:
    blob=f.read()

load_time=time.time()
graph=device.AllocateGraph(blob)

print('Time taken to load graph : {}'.format(time.time()-load_time))

total_time=time.time()

for _ in tqdm(range(int(2))):
    for path in paths:
        
        pp_time=time.time()
        img=get_image(path)
        p_time=time.time()
        graph.LoadTensor(img.astype(np.float16), 'user_object')
        import ipdb; ipdb.set_trace()
        output, _ = graph.GetResult()
        print('Time taken to preprocess : {}'.format(time.time()-pp_time))
        print('Prediction time : {}'.format(time.time()- p_time))
graph.DeallocateGraph()
device.CloseDevice()
