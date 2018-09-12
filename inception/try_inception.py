import tensorflow as tf
import numpy as np
import cv2
import os
import glob
import time

inception_save_dir=os.path.join('models', 'inception', 'inception-v3')

predictions=[]
outputs=[]
images=np.load(os.path.join('data', 'images', 'inception_images.npy'))

total_time=0

print('Loading Graph')
with tf.Session() as sess:
    start_time=time.time()
    saver=tf.train.import_meta_graph(inception_save_dir+'.meta')
    saver.restore(sess, inception_save_dir)
    graph=tf.get_default_graph()
    input_tensor=graph.get_tensor_by_name('input:0')
    output_tensor=graph.get_tensor_by_name('yo:0')
    print('Time Taken to Load Graph : {} secs'.format(time.time()-start_time))
    for i, img in enumerate(images):
        img=cv2.resize(img, (299,299))
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img=np.reshape(img, (1, 299, 299, 3))
        
        start_time=time.time()
        feed_dict={input_tensor:img}
        outputs.append(sess.run(output_tensor, feed_dict=feed_dict))
        one_time=time.time()-start_time
        print('Time taken for inference for image {} : {} secs'.format(i, one_time))
        total_time=total_time+one_time

print('Average time taken to perform inference for {} images is: {} seconds'.format(len(images), total_time/len(images)))
