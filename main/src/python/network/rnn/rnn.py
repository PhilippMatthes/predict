import tensorflow as tf
import numpy as np
import os
from functools import partial
from datetime import datetime

from main.src.python.oanda.oanda_date import OandaDate
from main.src.python.preparation.batch_manager import BatchManager
from main.src.python.config import models_path
from main.src.python.config import sessions_path
from main.src.python.config import predictions_path
from main.src.python.run_tensorboard import run_tensorboard

import matplotlib.pyplot as plt


class RNN:
    def __init__(self, logdir, n_inputs=1, n_steps=20, n_neurons=100, n_layers=3,
                 learning_rate=0.001, momentum=0.9, keep_prob=0.5,
                 name_scope="rnn", restore=False, store=True, training=True):
        self.name_scope = name_scope
        self.sess = tf.Session()

        self.store = store
        self.step = 0

        self.learning_rate = learning_rate
        self.keep_prob = keep_prob

        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = 1

        self.n_steps = n_steps

        with tf.name_scope(self.name_scope + "_base"):
            self.X = tf.placeholder(tf.float32, shape=[None, self.n_steps, self.n_inputs], name=self.name_scope + "X")
            self.y = tf.placeholder(tf.float32, shape=[None, self.n_steps, self.n_outputs], name=self.name_scope + "y")

        # Initialize layers
        with tf.name_scope(self.name_scope + "_network"):
            self.cells = [tf.contrib.rnn.LSTMCell(num_units=n_neurons, use_peepholes=True, activation=tf.nn.relu)
                          for _ in range(n_layers)]
            if training:
                self.cells = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob) for cell in self.cells]
            self.multi_layer_cell = tf.contrib.rnn.MultiRNNCell(self.cells)
            self.rnn_outputs, self.states = tf.nn.dynamic_rnn(cell=self.multi_layer_cell,
                                                              inputs=self.X,
                                                              dtype=tf.float32)
            self.stacked_rnn_outputs = tf.reshape(self.rnn_outputs, [-1, n_neurons])
            self.stacked_outputs = tf.layers.dense(self.stacked_rnn_outputs, self.n_outputs)
            self.outputs = tf.reshape(self.stacked_outputs, [-1, n_steps, self.n_outputs])

        # Initialize loss function
        with tf.name_scope(self.name_scope + "_loss"):
            self.loss = tf.reduce_mean(tf.square(self.outputs - self.y))

        # Initialize train function
        with tf.name_scope(self.name_scope + "_train"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.gvs = self.optimizer.compute_gradients(self.loss)
            self.capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gvs]
            self.training_op = self.optimizer.apply_gradients(self.capped_gvs)

        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        self.init = tf.global_variables_initializer()

        self.loss_train_summary = tf.summary.scalar(name_scope + "_Train_Loss", self.loss)

        if self.store:
            self.file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
        self.saver = tf.train.Saver()

        self.init.run(session=self.sess)
        if restore:
            self.saver.restore(self.sess, os.path.join(models_path, "model-{}.ckpt".format(self.name_scope)))

    def train(self, X, y, batch_size, n_epochs=100):
        print("Training network with batch size of {} and {} epochs".format(batch_size, n_epochs))
        batch_manager = BatchManager(X, y, batch_size)
        num_batches = batch_manager.num_batches()

        for epoch in range(n_epochs):
            print("Epoch {}/{}".format(epoch, n_epochs))
            while batch_manager.has_next():

                X_train, y_train = batch_manager.next()
                X_mini_batch_train = X_train.reshape(1, self.n_steps, 1)
                y_mini_batch_train = y_train.reshape(1, self.n_steps, 1)

                if batch_manager.i % int(num_batches / 50) == 0 and self.store:
                    summary_str_train = self.loss_train_summary.eval(
                        feed_dict={self.X: X_mini_batch_train, self.y: y_mini_batch_train},
                        session=self.sess,
                    )
                    # print(self.outputs.eval(
                    #    feed_dict={self.X: X_mini_batch_train},
                    #    session=self.sess
                    # ))
                    # print(y_mini_batch_train)
                    self.file_writer.add_summary(summary_str_train, self.step)
                    print("Feeding batch {}/{}".format(num_batches - batch_manager.i, num_batches))
                self.step += 1
                self.sess.run([self.training_op, self.extra_update_ops],
                              feed_dict={self.X: X_mini_batch_train, self.y: y_mini_batch_train})
            batch_manager.reset()
            if self.store:
                self.saver.save(self.sess, os.path.join(models_path, "model-{}.ckpt".format(self.name_scope)))

    def evaluate(self, X):
        outputs = self.sess.run([self.outputs], feed_dict={self.X: X})
        return np.array(outputs)

    def predict(self, input_sequence, iterations=300):
        if len(input_sequence) != self.n_steps:
            raise Exception("Input sequence length does not equal n_steps: {} != {}"
                            .format(len(input_sequence), self.n_steps))

        sequence = input_sequence.tolist()

        for iteration in range(iterations):
            X_batch = np.array(sequence[-self.n_steps:]).reshape(self.n_inputs, self.n_steps, self.n_outputs)
            y_pred = self.evaluate(X_batch)
            np.append(input_sequence, y_pred[0, -1, 0])
            sequence.append(y_pred[0, -1, 0])
            if iteration % 100 == 0:
                print("Being creative: iteration {}/{}".format(iteration, iterations))
        return sequence

    def tear_down(self):
        if self.store:
            self.file_writer.close()
        self.sess.close()
