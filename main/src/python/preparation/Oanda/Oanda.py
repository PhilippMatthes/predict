import configparser

import oandapy as opy
import pandas as pd

from main.src.python.preparation.Oanda.OandaDate import OandaDate

config = configparser.ConfigParser()
config.read('oanda.cfg')

oanda = opy.API(environment='practice', access_token=config['oanda']['access_token'])


class Oanda:
    @staticmethod
    def get_history_frame(instrument='EUR_USD', start=OandaDate().minus(days=5), end=OandaDate(), granularity='M1'):
        print(str(start))
        data = oanda.get_history(instrument=instrument,
                                 start=str(start),
                                 end=str(end),
                                 granularity=granularity)
        df = pd.DataFrame(data['candles']).set_index('time')
        df.index = pd.DatetimeIndex(df.index)
        return df
