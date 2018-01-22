from datetime import datetime

from main.src.python.download.interval import Interval
from main.src.python.download.reader import Reader
from main.src.python.helper.date_range_iterator import DateRangeIterator
from main.src.python.network.dnn import DNN
from main.src.python.oanda.oanda_date import OandaDate
from main.src.python.preparation.pca import PCA
from main.src.python.run_tensorboard import run_tensorboard

keep_variance = 0.99
learning_rate = 0.01
momentum = 0.9
data_offset = 64

start_date = OandaDate().with_date("2017-12-22 00:05:20")
end_date = OandaDate(datetime.utcnow())
batch_interval = Interval(days=5)

date_range_iterator = DateRangeIterator(start_date, end_date, batch_interval)

# PCA prefit
start, end = date_range_iterator.next()
reader = Reader(start=start, end=end)
data = reader.data
X_np = data[:-data_offset, 1:]
n_inputs = PCA.dimension_needed(X_np, variance=keep_variance)
pca = PCA(X=X_np, n_components=n_inputs)

dnn = DNN(n_inputs=n_inputs, n_outputs=data_offset, learning_rate=0.01, keep_variance=0.99, momentum=0.9,
          name_scope="dnn")

while date_range_iterator.has_next():
    try:
        start, end = date_range_iterator.next()

        print("Parsing date range: {} to {}".format(start.description(), end.description()))
        reader = Reader(start=start, end=end)
        data = reader.data

        X_np = data[:-data_offset, 1:]
        y_np = data[data_offset:, :1]

        pca.refit(X_np)
        X_np_reduced = pca.reduce(X_np)

        dnn.train(X_np_reduced, y_np, batch_size=100, n_epochs=10)

    except KeyboardInterrupt:
        dnn.tear_down()

run_tensorboard()



