import tensorflow as tf
import numpy as np
import os
from functools import partial
from datetime import datetime

from main.src.python.download.reader import Reader
from main.src.python.preparation.batch_manager import BatchManager
from main.src.python.preparation.pca import PCA
from main.src.python.config import models_path
from main.src.python.config import sessions_path
from main.src.python.run_tensorboard import run_tensorboard

now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
logdir = os.path.join(sessions_path, "run-{}/".format(now))

data_offset = int(600 / 5)
batch_size = 100
n_epochs = 1000
learning_rate = 0.01

# Read data
reader = Reader(read=True, scale=True)
data = reader.data

# Truncate data
X_np = data[:-data_offset, 1:]
y_np = data[data_offset:, :1]

# Check how many dimensions we need to fit the given variance
n_inputs = PCA.dimension_needed(X_np, variance=0.97)

# Perform dimension reduction
pca = PCA(X=X_np, n_components=n_inputs)
X_np_reduced = pca.reduce(X=X_np)

# Load batch manager
batch_manager = BatchManager(X=X_np_reduced, y=y_np, batch_size=batch_size)

n_hidden1 = data_offset * 4
n_hidden2 = data_offset * 3
n_hidden3 = data_offset * 2
n_outputs = data_offset

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.float32, shape=(None), name="y")

# Training placeholder to turn off batch norm out of training
training = tf.placeholder_with_default(False, shape=(), name="training")

batch_norm_layer = partial(tf.layers.batch_normalization, training=training, momentum=0.9)

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1")
    bn1 = batch_norm_layer(hidden1)
    bn1_act = tf.nn.elu(bn1)
    hidden2 = tf.layers.dense(bn1_act, n_hidden2, name="hidden2")
    bn2 = batch_norm_layer(hidden2)
    bn2_act = tf.nn.elu(bn2)
    hidden3 = tf.layers.dense(bn2_act, n_hidden3, name="hidden3")
    bn3 = batch_norm_layer(hidden3)
    bn3_act = tf.nn.elu(bn3)
    predictions_before_bn = tf.layers.dense(bn3_act, n_outputs, name="outputs")
    predictions = batch_norm_layer(predictions_before_bn)

with tf.name_scope("loss"):
    error = predictions - y
    mse = tf.reduce_mean(tf.square(error), name="mse")


with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

mse_train_summary = tf.summary.scalar("MSE-Train", mse)
mse_test_summary = tf.summary.scalar("MSE-Test", mse)

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

try:
    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            num_batches = batch_manager.num_batches()
            while batch_manager.has_next():
                X_train, X_test, y_train, y_test = batch_manager.next()
                if batch_manager.i % int(num_batches / 50) == 0:
                    summary_str_train = mse_train_summary.eval(feed_dict={training: False, X: X_train, y: y_train})
                    summary_str_test = mse_test_summary.eval(feed_dict={training: False, X: X_test, y: y_test})
                    mse_train = mse.eval(feed_dict={training: False, X: X_train, y: y_train})
                    mse_test = mse.eval(feed_dict={training: False, X: X_test, y: y_test})
                    step = epoch * num_batches - batch_manager.i
                    print("Epoch: {}/{} Step {} Train Loss: {}, Test Loss: {} (Discrepancy: {})"
                          .format(epoch, n_epochs, step,  mse_train, mse_test, mse_train-mse_test))
                    file_writer.add_summary(summary_str_train, step)
                    file_writer.add_summary(summary_str_test, step)
                sess.run([training_op, extra_update_ops], feed_dict={training: True, X: X_train, y: y_train})
            batch_manager.reset()
        save_path = saver.save(sess, os.path.join(models_path, "dnn.ckpt"))

except KeyboardInterrupt:
    pass

file_writer.close()
run_tensorboard()
