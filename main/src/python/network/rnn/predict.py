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

learning_rate = 0.01
momentum = 0.9

restore = True

batch_size = 50
n_test_size = 50
shift = 1

iterations = 1000

instrument = "EUR_USD"

start = OandaDate().with_date("2018-01-15 16:40:00")
end = OandaDate().with_date("2018-01-19 22:00:00")

now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
logdir = os.path.join(sessions_path, "run-{}/".format(now))

rnn = RNN(logdir=os.path.join(sessions_path, "rnn-run-{}/".format(now)),
          n_inputs=1,
          n_steps=batch_size,
          n_neurons=100,
          n_layers=3,
          learning_rate=0.001,
          momentum=0.9,
          keep_prob=0.5,
          name_scope="rnn",
          restore=True,
          store=False,
          training=False)

# scaler = StandardScaler()

try:
    print("Parsing date range: {} to {}".format(start.description(), end.description()))
    reader = SerialReader(instrument, start, end, reduce=True)
    data = reader.data

    row = 0

    X_np_serial = np.diff(data[:-shift, row])
    y_np_serial = np.diff(data[shift:, row])

    length = X_np_serial.shape[0] - X_np_serial.shape[0] % shift

    X_np_truncated = X_np_serial[:length]
    y_np_truncated = y_np_serial[:length]

    height = int(length / shift)

    X_stacked = np.reshape(X_np_truncated, (height, shift))
    y_stacked = np.reshape(y_np_truncated, (height, shift))

    # scaler.partial_fit(X_stacked, y_stacked)

    cutoff_X = len(X_stacked) % (batch_size + n_test_size)
    cutoff_y = len(y_stacked) % (batch_size + n_test_size)
    X_feed = X_stacked[cutoff_X:]
    y_feed = y_stacked[cutoff_y:]

    y_predicted = rnn.predict(X_feed[:batch_size], iterations=iterations)

    plt.plot(np.reshape(y_predicted, (-1, 1))[:iterations])
    plt.plot(np.reshape(y_feed, (-1, 1))[:iterations], alpha=0.5)

    plt.show()

except KeyboardInterrupt:
    rnn.tear_down()
