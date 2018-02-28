import threading
from datetime import datetime
import matplotlib.pyplot as plt
import os

import numpy as np

from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from main.src.python.download.interval import Interval
from main.src.python.download.serial_reader import SerialReader
from main.src.python.helper.date_range_iterator import DateRangeIterator
from main.src.python.network.rnn.rnn import RNN
from main.src.python.oanda.oanda_date import OandaDate
from main.src.python.run_tensorboard import run_tensorboard
from main.src.python.config import pca_path, sessions_path
from main.src.python.config import scaler_path
from main.src.python.helper.flatten import flatten

keep_variance = 0.99
momentum = 0.9

shift = 1

batch_size = 50
n_test_size = 50
n_epochs = 1

instrument = "EUR_USD"

start_date = OandaDate().with_date("2017-09-07 15:00:00")
end_date = OandaDate(datetime.utcnow())
batch_interval = Interval(days=5)

date_range_iterator = DateRangeIterator(start_date, end_date, batch_interval)

now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

rnn = RNN(logdir=os.path.join(sessions_path, "rnn-run-{}/".format(now)),
          n_inputs=1,
          n_steps=batch_size,
          n_neurons=100,
          n_layers=10,
          learning_rate=0.01,
          momentum=0.9,
          keep_prob=0.5,
          name_scope="rnn",
          restore=True,
          store=True,
          training=True)
# scaler = StandardScaler()

while date_range_iterator.has_next():
    try:
        start, end = date_range_iterator.next()

        print("Parsing date range: {} to {}".format(start.description(), end.description()))
        reader = SerialReader(instrument, start, end, reduce=True)
        data = reader.data

        row = 0
        # Shift input data
        X_np_serial = np.diff(data[:-shift, row])
        y_np_serial = np.diff(data[shift:, row])

        length = X_np_serial.shape[0] - X_np_serial.shape[0] % shift

        X_np_truncated = X_np_serial[:length]
        y_np_truncated = y_np_serial[:length]

        height = int(length / shift)

        X_stacked = np.reshape(X_np_truncated, (height, shift))
        y_stacked = np.reshape(y_np_truncated, (height, shift))

        # scaler.partial_fit(X_stacked, y_stacked)
        # joblib.dump(scaler, os.path.join(scaler_path, "scaler-{}.pickle".format(instrument)))

        cutoff_X = len(X_stacked) % (batch_size + n_test_size)
        cutoff_y = len(y_stacked) % (batch_size + n_test_size)
        X_feed = X_stacked[cutoff_X:]
        y_feed = y_stacked[cutoff_y:]

        rnn.train(X_feed, y_feed, batch_size, n_epochs)

    except KeyboardInterrupt:
        rnn.tear_down()
