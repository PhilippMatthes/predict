import threading
from datetime import datetime
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from main.src.python.download.interval import Interval
from main.src.python.download.parallel_reader import Reader
from main.src.python.helper.date_range_iterator import DateRangeIterator
from main.src.python.network.dnn import DNN
from main.src.python.oanda.oanda_date import OandaDate
from main.src.python.run_tensorboard import run_tensorboard
from main.src.python.config import pca_path, sessions_path
from main.src.python.config import scaler_path

keep_variance = 0.99
learning_rate = 0.01
momentum = 0.9

chunk_size = 64

restore = True

batch_size = 100
n_epochs = 100

instruments = Reader.all_instruments()
instruments = ["EUR_USD"]

start = OandaDate().with_date("2018-01-15 16:40:00")
end = OandaDate().with_date("2018-01-19 22:00:00")

networks = {}

now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
logdir = os.path.join(sessions_path, "run-{}/".format(now))

dnn = DNN(n_inputs=chunk_size,
          n_outputs=chunk_size,
          logdir=logdir,
          learning_rate=learning_rate,
          keep_variance=keep_variance,
          momentum=momentum,
          name_scope="dnn-pretrain",
          restore=True,
          store=False)

scaler = StandardScaler()
try:
    print("Parsing date range: {} to {}".format(start.description(), end.description()))
    reader = Reader(start, end, reduce=True)
    data = reader.data

    for instrument in instruments:
        row = instruments.index(instrument)
        # Shift input data
        X_np_serial = data[:, row]
        y_np_serial = data[chunk_size:, row]

        X_length = X_np_serial.shape[0] - X_np_serial.shape[0] % chunk_size
        y_length = y_np_serial.shape[0] - y_np_serial.shape[0] % chunk_size

        X_np_truncated = X_np_serial[:X_length]
        y_np_truncated = y_np_serial[:y_length]

        X_height = int(X_length / chunk_size)
        y_height = int(y_length / chunk_size)

        X_stacked = np.reshape(X_np_truncated, (X_height, chunk_size))
        y_stacked = np.reshape(y_np_truncated, (y_height, chunk_size))

        y_predicted = dnn.predict(X_stacked)

        plt.plot(np.reshape(y_predicted, (X_length, 1)))
        plt.plot(np.reshape(y_np_serial, (y_length, 1)))

        plt.show()

except KeyboardInterrupt:
    for network in networks.values():
        network.tear_down()
