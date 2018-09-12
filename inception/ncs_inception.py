import numpy as np
import mvnc.mvncapi as mvnc
import os
import cv2
import time

images=np.load(os.path.join('data', 'images', 'inception_images.npy'))

inception_graph_path=os.path.join('models', 'ncs_inception', 'inception_inference.graph')

devices=mvnc.EnumerateDevices()

if len(devices)==0:
    print('No Devices Found')
    quit()

device=mvnc.Device(devices[0])

device.OpenDevice()

with open(inception_graph_path, mode='rb') as f:
    inception_blob=f.read()

start_time=time.time()

inception_graph=device.AllocateGraph(inception_blob)

total_time=time.time()-start_time
print('Total Time to Load graph: {} secs'.format(total_time))

start_time=time.time()
for image in images:
    img=cv2.resize(image, (299,299))
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=np.reshape(img, (1, 299, 299, 3))
    inception_graph.LoadTensor(img.astype(np.float32), 'user object')
    output, user_obj=inception_graph.GetResult()

total_time=time.time()-start_time
print('Total Time for 24 images: {} secs'.format(total_time))
print('Average Time for image: {} secs'.format(total_time/24))

inception_graph.DeallocateGraph()
device.CloseDevice()
