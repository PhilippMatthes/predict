import configparser
from json import JSONDecodeError

import oandapy as opy
import pandas as pd

from main.src.python.download.interval import Interval
from main.src.python.oanda.oanda_granularity import OandaGranularity

from main.src.python.oanda.oanda_date import OandaDate

max_data = 5000
data_padding = 500

class Oanda:
    @staticmethod
    def get_history_frame(instrument='EUR_USD',
                          start=OandaDate().minus(minutes=1),
                          end=OandaDate(),
                          granularity=OandaGranularity.s5["api"]):
        config = configparser.ConfigParser()
        config.read('oanda.cfg')
        oanda = opy.API(environment='practice',
                        access_token=config['oanda']['access_token'])
        try:
            data = oanda.get_history(instrument=instrument,
                                     start=str(start),
                                     end=str(end),
                                     granularity=granularity,
                                     price="M")
            data_frame = pd.DataFrame(data['candles']).set_index('time')
            data_frame.index = pd.DatetimeIndex(data_frame.index)
            return data_frame

        except JSONDecodeError:
            print("No Data received")

    @staticmethod
    def range(start=OandaDate(), end=OandaDate(), interval=Interval(seconds=OandaGranularity.s5["sec"])):
        possible_request_data = max_data - data_padding
        possible_end = start.plus(seconds=(possible_request_data*interval.in_seconds()))
        if possible_end.is_after(end):
            return start, end
        else:
            return start, possible_end


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    from time import sleep

    plt.ion()

    try:
        while True:
            history_frame = Oanda.get_history_frame(start=OandaDate().minus(minutes=1))

            print(history_frame)

            y = history_frame["closeAsk"]
            x = y.index.tolist()

            plt.clf()
            plt.plot(x, np.gradient(y))

            plt.gcf().autofmt_xdate()

            plt.pause(5)
            sleep(5)
            plt.draw()
    except KeyboardInterrupt:
        pass
