import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import time
inception_save_dir=os.path.join('resources', 'inference_mnist_model', 'model')

predictions=[]
outputs=[]
images=np.load(os.path.join('resources', 'mnist_images.npy'))

print('Loading Graph')
with tf.Session() as sess:
    start_time=time.time()
    saver=tf.train.import_meta_graph(inception_save_dir+'.meta')
    saver.restore(sess, inception_save_dir)
    graph=tf.get_default_graph()
    input_tensor=graph.get_tensor_by_name('input:0')
    output_tensor=graph.get_tensor_by_name('output:0')
    print('Time Taken to Load Graph = {}'.format(time.time()-start_time))
    for img in images:
        #img=cv2.resize(img, (299,299))
        #img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #img=np.reshape(img, (1, 299, 299, 3))
        img=np.reshape(img, (1, 784))
        start_time=time.time()
        feed_dict={input_tensor:img}
        outputs.append(sess.run(output_tensor, feed_dict=feed_dict))
        total_time=time.time()-start_time

print('Time taken to perform inference : {}'.format(total_time/len(images)))
