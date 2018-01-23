import tensorflow as tf
import numpy as np
import os
from functools import partial
from datetime import datetime

from main.src.python.download.reader import Reader
from main.src.python.oanda.oanda_date import OandaDate
from main.src.python.preparation.batch_manager import BatchManager
from main.src.python.config import models_path
from main.src.python.config import sessions_path
from main.src.python.run_tensorboard import run_tensorboard


class DNN:
    def __init__(self, n_inputs, n_outputs, logdir, learning_rate=0.01, keep_variance=0.99, momentum=0.9, name_scope="dnn",
                 restore=False, store=True):
        self.name_scope = name_scope
        self.sess = tf.Session()

        self.store = store
        self.step = 0

        self.learning_rate = learning_rate
        self.keep_variance = keep_variance

        self.n_inputs = n_inputs
        n_hidden1 = 512*4
        n_hidden2 = 256*4
        n_hidden3 = 128*4
        self.n_outputs = n_outputs

        self.training = tf.placeholder_with_default(False, shape=(), name=self.name_scope+"training")
        self.X = tf.placeholder(tf.float32, shape=(None, n_inputs), name=self.name_scope+"X")
        self.y = tf.placeholder(tf.float32, shape=(None), name=self.name_scope+"y")

        batch_norm_layer = partial(tf.layers.batch_normalization, training=self.training, momentum=momentum)

        # Initialize layers
        with tf.name_scope(self.name_scope):
            self.hidden1 = tf.layers.dense(self.X, n_hidden1, name=self.name_scope+"_hidden1")
            self.bn1 = batch_norm_layer(self.hidden1)
            self.bn1_act = tf.nn.elu(self.bn1)
            self.hidden2 = tf.layers.dense(self.bn1_act, n_hidden2, name=self.name_scope+"_hidden2")
            self.bn2 = batch_norm_layer(self.hidden2)
            self.bn2_act = tf.nn.elu(self.bn2)
            self.hidden3 = tf.layers.dense(self.bn2_act, n_hidden3, name=self.name_scope+"_hidden3")
            self.bn3 = batch_norm_layer(self.hidden3)
            self.bn3_act = tf.nn.elu(self.bn3)
            self.predictions_before_bn = tf.layers.dense(self.bn3_act, n_outputs, name=self.name_scope+"_outputs")
            self.predictions = batch_norm_layer(self.predictions_before_bn)

        # Initialize loss function
        with tf.name_scope(self.name_scope + "_loss"):
            self.mse = tf.losses.mean_squared_error(labels=self.y, predictions=self.predictions)

        # Initialize train function
        with tf.name_scope(self.name_scope + "_train"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate)
            self.training_op = self.optimizer.minimize(self.mse)

        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        self.init = tf.global_variables_initializer()

        self.mse_train_summary = tf.summary.scalar(name_scope + "_Train_MSE", self.mse)
        self.mse_test_summary = tf.summary.scalar(name_scope + "_Test_MSE", self.mse)

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
            while batch_manager.has_next():
                X_train, X_test, y_train, y_test = batch_manager.next()
                if batch_manager.i % int(num_batches / 50) == 0 and self.store:
                    summary_str_train = self.mse_train_summary.eval(
                        feed_dict={self.training: False, self.X: X_train, self.y: y_train},
                        session=self.sess,
                    )
                    summary_str_test = self.mse_test_summary.eval(
                        feed_dict={self.training: False, self.X: X_test, self.y: y_test},
                        session=self.sess,
                    )
                    self.file_writer.add_summary(summary_str_train, self.step)
                    self.file_writer.add_summary(summary_str_test, self.step)
                self.step += 1
                self.sess.run([self.training_op, self.extra_update_ops],
                              feed_dict={self.training: True, self.X: X_train, self.y: y_train})
            batch_manager.reset()
            if self.store:
                self.saver.save(self.sess, os.path.join(models_path, "model-{}.ckpt".format(self.name_scope)))

    def predict(self, X):
        return self.predictions.eval(session=self.sess, feed_dict={self.training: False, self.X: X})

    def tear_down(self):
        if self.store:
            self.file_writer.close()
        self.sess.close()


