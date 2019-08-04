#!/usr/bin/env python

'''
Adapted from https://github.com/huseinzol05/NLP-Models-Tensorflow/blob/master/text-classification/65.transfer-learning-gpt2.ipynb
'''

import time
import model, encoder
from accumulate import AccumulatingOptimizer
import tensorflow as tf
import json

from utils import *
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from keras.preprocessing.sequence import pad_sequences

import nltk
nltk.download('stopwords')

class Model:
    def __init__(self, dimension_output, learning_rate = 0.0001):
        self.X = tf.placeholder(tf.int32, [None, None])
        self.Y = tf.placeholder(tf.int32, [None])
        output = model.model(hparams=hparams, X=self.X)['logits']
        output = tf.reduce_mean(output, axis = 1)
        self.logits = tf.layers.dense(output, dimension_output)
        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits = self.logits, labels = self.Y
            )
        )
        self.all_vars = [v for v in tf.trainable_variables() if 'model' in v.name]
        train_vars = [v for v in self.all_vars if '/h' in v.name]
#         opt = AccumulatingOptimizer(
#             opt=tf.train.AdamOptimizer(learning_rate=learning_rate),
#             var_list=train_vars)
#         opt_reset = opt.reset()
#         opt_compute = opt.compute_gradients(self.cost)
#         self.optimizer = opt.apply_gradients()

        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
        
        print(output)
        
        correct_pred = tf.equal(
            tf.argmax(self.logits, 1, output_type = tf.int32), self.Y
        )
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

if __name__ == '__main__':

    model_name = '345M'

    enc = encoder.get_encoder(model_name, 'models')
    hparams = model.default_hparams()

    with open(f'models/{model_name}/hparams.json') as f:
        hparams.override_from_dict(json.load(f))

    print(enc.encode('this is a test'))

    dimension_output = 2
    learning_rate = 0.0001

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    modelnn = Model(
        dimension_output,
        learning_rate
    )

    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(var_list = modelnn.all_vars)
    saver.restore(sess, f'models/{model_name}/model.ckpt')

    trainset = sklearn.datasets.load_files(container_path = 'data', encoding = 'UTF-8')
    trainset.data, trainset.target = separate_dataset(trainset,1.0)
    print (trainset.target_names)
    print (len(trainset.data))
    print (len(trainset.target))

    maxlen = 100
    batch_size = 16



    X = []
    for text in tqdm(trainset.data):
        X.append(enc.encode(text)[:maxlen])


    X = pad_sequences(X, padding='post')
    print(X.shape)


    train_X, test_X, train_Y, test_Y = train_test_split(
        X, trainset.target, test_size = 0.2
    )



    EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 3, 0, 0, 0

    while True:
        lasttime = time.time()
        if CURRENT_CHECKPOINT == EARLY_STOPPING:
            print('break epoch:%d\n' % (EPOCH))
            break

        train_acc, train_loss, test_acc, test_loss = 0, 0, 0, 0
        pbar = tqdm(
            range(0, len(train_X), batch_size), desc = 'train minibatch loop'
        )
        for i in pbar:
            index = min(i + batch_size, len(train_X))
            batch_x = train_X[i: index]
            batch_y = train_Y[i: index]
            acc, cost, _ = sess.run(
                [modelnn.accuracy, modelnn.cost, modelnn.optimizer],
                feed_dict = {
                    modelnn.Y: batch_y,
                    modelnn.X: batch_x
                },
            )
            assert not np.isnan(cost)
            train_loss += cost
            train_acc += acc
            pbar.set_postfix(cost = cost, accuracy = acc)
        
        pbar = tqdm(range(0, len(test_X), batch_size), desc = 'test minibatch loop')
        for i in pbar:
            index = min(i + batch_size, len(test_X))
            batch_x = test_X[i: index]
            batch_y = test_Y[i: index]
            acc, cost = sess.run(
                [modelnn.accuracy, modelnn.cost],
                feed_dict = {
                    modelnn.Y: batch_y,
                    modelnn.X: batch_x,
                },
            )
            test_loss += cost
            test_acc += acc
            pbar.set_postfix(cost = cost, accuracy = acc)

        train_loss /= len(train_X) / batch_size
        train_acc /= len(train_X) / batch_size
        test_loss /= len(test_X) / batch_size
        test_acc /= len(test_X) / batch_size

        if test_acc > CURRENT_ACC:
            print(
                'epoch: %d, pass acc: %f, current acc: %f'
                % (EPOCH, CURRENT_ACC, test_acc)
            )
            CURRENT_ACC = test_acc
            CURRENT_CHECKPOINT = 0
        else:
            CURRENT_CHECKPOINT += 1
            
        print('time taken:', time.time() - lasttime)
        print(
            'epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n'
            % (EPOCH, train_loss, train_acc, test_loss, test_acc)
        )
        EPOCH += 1