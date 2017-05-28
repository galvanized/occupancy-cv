# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import numpy as np

import time
from PIL import Image
import subprocess
import os
import glob

epochs = 10000
batch_size = 100

data_file = 'data/data.npz'
model_dir = 'data/model/'
model_name = 'model'

FLAGS = None

class ImageData():
    def __init__(self, filename, test_f=0.2):
        self.raw_data = np.load(filename)
        self.x_dims = len(self.raw_data['xs'][0])
        self.y_dims = len(self.raw_data['ys'][0])
        self.size = len(self.raw_data['xs'])
        self.test_f = test_f #test fraction
        self.test_size = int(self.size*self.test_f)
        self.train_size = self.size - self.test_size

        test_indices = []
        print("Generating indices.")
        while len(test_indices) < self.test_size:
            r = -1
            while r in test_indices or r==-1:
                r = np.random.randint(self.size)
            test_indices.append(r)
        print("Remapping.")

        self.rxs = self.raw_data['xs']
        self.rys = self.raw_data['ys']

        del(self.raw_data)

        self.xs = np.empty(shape=(self.train_size, self.x_dims), dtype='uint8')
        self.ys = np.empty(shape=(self.train_size, self.y_dims), dtype='uint8')

        self.txs = np.empty(shape=(self.test_size, self.x_dims), dtype='uint8')
        self.tys = np.empty(shape=(self.test_size, self.y_dims), dtype='uint8')

        

        train_index = 0
        test_index = 0
        for full_index in range(self.size):
            # todo: make up mind on whether to spend time but save memory by using raw_data
            if full_index in test_indices:
                self.txs[test_index] = self.rxs[full_index]
                self.tys[test_index] = self.rys[full_index]
                test_index += 1
            else:
                self.xs[train_index] = self.rxs[full_index]
                self.ys[train_index] = self.rys[full_index]
                train_index += 1

        del(self.rxs, self.rys)

        print('Import successful.', self.size, 'items.')

    def get_batch(self, quantity):
        xs = []
        ys = []
        for i in np.random.randint(self.train_size, size=quantity):
            xs.append(self.xs[i])
            ys.append(self.ys[i])
        return xs, ys




def main(_):
    
    data = ImageData(data_file)

    # inputs : placeholder is substituted by computer
    x = tf.placeholder(tf.float32, [None, data.x_dims]) # any number of items (dimension of unknown length) with 784 px/dimensions
    # weights : variable is modified by computer
    W = tf.Variable(tf.zeros([data.x_dims, data.y_dims])) # 784 each with 10 solution dimensions
    # biases
    b = tf.Variable(tf.zeros([data.y_dims])) # added to output, matches dimension (10x1)
    # model (value of output layer)
    y = tf.matmul(x, W) + b

    tf.add_to_collection("y", y)

    # define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 2]) # output placeholder used by computer

    # loss function
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    # training function
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(cross_entropy) # learning rate of .5

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # train (stochastic gradient descent - 100 at a time for frugality)
        for _ in range(epochs):
            if _%(epochs/100) == 0:
                print(int(_/epochs * 100), '%')
            batch_xs, batch_ys = data.get_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(sess.run(accuracy, feed_dict={x: data.txs,
            y_: data.tys}))

        '''
        os.makedirs(model_dir, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, model_dir+model_name)
        saver.export_meta_graph(model_dir+model_name+'.meta')
        '''

        #print(sess.run(y, feed_dict={x: [data.txs[9]]}))
        #print([data.xs[0]])
        #tf.train.export_meta_graph(filename='photomodel.meta')


        command = "exec cvlc -I dummy v4l2:///dev/video1 --video-filter scene --no-audio --scene-path . --scene-prefix temp- --scene-format png --run-time=1"

        output_resolution = (15,10) #(52,39)

        process = subprocess.Popen([command], shell=True)

        while 1:

            
            time.sleep(0.2)

            for filename in glob.glob('temp*.png'):
                im = Image.open(filename)
                #im = Image.open(input("Filename: "))
                os.remove(filename)
                om = im.resize(output_resolution)
                l = list(om.getdata())
                channels = 3
                nx = []
                for i, p in enumerate(l):
                    for c in range(channels):
                        nx.append(p[c])

                time.sleep(1)
                result = sess.run(y, feed_dict={x: [nx]})
                print(result)
                print("Here!" if result[0][0] > result[0][1] else "Away.")
                print()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
