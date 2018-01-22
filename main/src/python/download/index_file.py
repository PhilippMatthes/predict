from main.src.python.oanda.oanda_date import OandaDate
import pandas as pd

date_length = len("2017-12-17 22:50:05")
ending_length = len(".pickle")
separator_length = len("_to_")


class IndexFile:
    def __init__(self, path):
        self.path = path
        self.start = OandaDate().with_date(path[-(date_length+date_length+ending_length+separator_length):-(date_length+ending_length+separator_length)])
        self.end = OandaDate().with_date(path[-(date_length+ending_length):-ending_length])

    def needed_between(self, start=OandaDate(), end=OandaDate()):
        before = self.start.is_before(start) and self.end.is_after(start)
        in_range = self.start.is_after(start) and self.end.is_before(end)
        after = self.start.is_before(end) and self.end.is_after(end)
        return before or in_range or after

    def read(self):
        return pd.read_pickle(self.path)

    def read_between(self, start=OandaDate(), end=OandaDate()):
        if start.is_after(end):
            raise Exception("Start needs to be before end!")
        df = pd.read_pickle(self.path)
        start_dt = start.date.date()
        end_dt = end.date.date()
        return df[(df.index.date > start_dt) | (df.index.date < end_dt)]
