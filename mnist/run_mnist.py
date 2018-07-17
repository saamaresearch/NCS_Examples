import mvnc.mvncapi as mvnc
import os
import numpy as np
import time

graph_path=os.path.join('resources', 'inference_mnist_model', 'mnist_inference_2.graph')
images=np.load(os.path.join('resources', 'mnist_images.npy'))

#Check if Device is Present. Quit Otherwise
devices=mvnc.EnumerateDevices()

if len(devices)==0:
    print('No Devices Found')
    quit()

#If device is present, then open it
device=mvnc.Device(devices[0])
device.OpenDevice()

#Read the graph file and load it into the device
with open(graph_path, mode='rb') as f:
    blob=f.read()

start_time=time.time()
graph=device.AllocateGraph(blob)
print('Time taken to load graph = {}'.format(time.time()-start_time))

total_time=0

#Perform Images on a few saved mnist images
for image in images:
    img=np.reshape(image, (1, 784))
    #Load the image into the NCS for inference
    start_time=time.time()
    graph.LoadTensor(img.astype(np.float16), 'user_object')
    #Get the result
    output, usr_obj=graph.GetResult()
    total_time+=(time.time()-start_time)
    #print(np.argmax(output))

print('Time take to perform inference = {}'.format(total_time/len(images)))
#Deallocate the graph from the NCS device
graph.DeallocateGraph()

#Close the Device
device.CloseDevice()
