import threading
from datetime import datetime
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

from main.src.python.download.interval import Interval
from main.src.python.download.reader import Reader
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

restore = False
store = True

batch_size = 100
n_epochs = 100

instruments = Reader.all_instruments()
instruments = ["AUD_CAD"]

start_date = OandaDate().with_date("2017-11-07 00:00:00")
end_date = OandaDate(datetime.utcnow())
batch_interval = Interval(days=5)

date_range_iterator = DateRangeIterator(start_date, end_date, batch_interval)

networks = {}

now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
logdir = os.path.join(sessions_path, "run-{}/".format(now))

for instrument in instruments:
    networks[instrument] = DNN(n_inputs=chunk_size,
                               n_outputs=chunk_size,
                               logdir=logdir,
                               learning_rate=learning_rate,
                               keep_variance=keep_variance,
                               momentum=momentum,
                               name_scope="dnn-{}".format(instrument),
                               restore=restore,
                               store=store)

scaler = StandardScaler()

while date_range_iterator.has_next():
    try:
        start, end = date_range_iterator.next()

        print("Parsing date range: {} to {}".format(start.description(), end.description()))
        reader = Reader(start, end, reduce=False)
        data = reader.data

        threads = []

        for instrument in instruments:
            row = instruments.index(instrument)
            # Shift input data
            X_np_serial = data[:-chunk_size, row]
            y_np_serial = data[chunk_size:, row]

            length = X_np_serial.shape[0] - X_np_serial.shape[0] % chunk_size

            X_np_truncated = X_np_serial[:length]
            y_np_truncated = y_np_serial[:length]

            height = int(length / chunk_size)

            X_stacked = np.reshape(X_np_truncated, (height, chunk_size))
            y_stacked = np.reshape(y_np_truncated, (height, chunk_size))

            scaler.partial_fit(X_stacked, y_stacked)
            X_scaled = scaler.transform(X_stacked)
            y_scaled = scaler.transform(y_stacked)

            print(X_stacked.shape, y_stacked.shape)

            dnn = networks[instrument]
            thread = threading.Thread(target=dnn.train, args=[X_scaled, y_scaled, batch_size, n_epochs])
            thread.daemon = True
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    except KeyboardInterrupt:
        for network in networks.values():
            network.tear_down()
