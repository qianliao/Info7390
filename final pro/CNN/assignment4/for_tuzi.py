import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm
from read_ubyte import load_mnist_test, load_mnist_train

def weight_initializer(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_initializer(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

def conv1d(x, W):
    return tf.nn.conv1d(x, W, 1, padding='SAME')

def max_pool_2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1],
                        strides=[1, 2, 1], padding='SAME')


if __name__ == '__main__':

    with tf.Graph().as_default():

        x = tf.placeholder("float", shape=[None, 6,1])
        p_true = tf.placeholder("float", shape=[None, 2])

        W_conv1 = weight_initializer([2,1,4])
        b_conv1 = bias_initializer([4])

        h_conv1 = tf.nn.relu(conv1d(x, W_conv1)+ b_conv1)
        # h_pool1 = max_pool_2(h_conv1)

        W_fc1 = weight_initializer([6*4, 32])
        b_fc1 = bias_initializer([32])

        flat = tf.reshape(h_conv1, [-1,6*4])
        h_fc1 = tf.nn.relu(tf.matmul(flat, W_fc1) + b_fc1)

        W_fc2 = weight_initializer([32,2])
        b_fc2 = bias_initializer([2])

        p_pred = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

        loss = tf.reduce_sum(p_true* tf.log(tf.clip_by_value(p_pred,1e-12,1.0)))
        cross_entropy = -loss
        # cross_entropy = -tf.reduce_sum(p_true* tf.log(p_pred+1e-5))

        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        correct_predict = tf.equal(tf.argmax(p_true,1), tf.argmax(p_pred,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))

        # tf.summary.scalar('loss', loss)
        
        #Load data
        data_train = np.random.rand(1000, 6, 1)
        y = np.random.randint(2, size=(1000))
        label_train = np.zeros((1000,2))
        for i in xrange(1000):
            if (y[i] == 0):
                label_train[i][0] = 1
            else:
                label_train[i][1] = 1

        # data_train,label_train = load_mnist_train()
        # data_test,label_test = load_mnist_test()
        
        # summary = tf.summary.merge_all()

        # sess = tf.InteractiveSession()
        sess = tf.Session()
        # summary_writer = tf.summary.FileWriter('mnist_logs', sess.graph)
        sess.run(tf.global_variables_initializer())
        for e in xrange(50):
            print ("epoch:%s"%e)
            sess.run(train_step, feed_dict = {x:data_train, p_true:label_train})
                
            acc = sess.run(accuracy, feed_dict = {x:data_train, p_true:label_train})
            print "accracy_val:", acc
