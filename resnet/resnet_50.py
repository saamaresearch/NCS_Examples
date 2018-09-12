import numpy as np
import tensorflow as tf

from tensorflow.contrib.slim.nets import resnet_v1

slim = tf.contrib.slim

import ipdb; ipdb.set_trace()

def run(name, image_size, num_classes):
  with tf.Graph().as_default():
    image = tf.placeholder("float", [1, image_size, image_size, 3], name="input")
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits, _ = resnet_v1.resnet_v1_101(image, num_classes, is_training=False)
    probabilities = tf.nn.softmax(logits)
    init_fn = slim.assign_from_checkpoint_fn('resnet_v1_101.ckpt', slim.get_model_variables('resnet_v1'))

    with tf.Session() as sess:
        init_fn(sess)
        saver = tf.train.Saver(tf.global_variables())
        saver.save(sess, "output/"+name)

run('resnet-50', 224, 1000)
