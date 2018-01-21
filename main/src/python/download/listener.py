from datetime import datetime

from main.src.python.oanda.oanda_date import OandaDate
from main.src.python.oanda.oanda_granularity import OandaGranularity
from main.src.python.download.frame_queue import FrameQueue
from main.src.python.oanda.oanda import Oanda

import pandas as pd

import threading
from time import sleep

from main.src.python.download.interval import Interval
from main.src.python.download.reader import Reader

cycle_duration = 360


class Listener:
    def __init__(self):
        self.frame_queues = {}
        self.stop_flag = False

    def start_polling(self, instrument='EUR_USD', interval=Interval(seconds=OandaGranularity.s5["sec"]),
                      granularity=OandaGranularity.s5["api"]):
        self.frame_queues[instrument] = FrameQueue(instrument=instrument)
        thread = threading.Thread(target=self.poll, args=[instrument, interval, granularity])
        thread.daemon = True
        thread.start()

    def start_building_history(self, instrument, start, end, granularity_tuple):
        self.frame_queues[instrument] = FrameQueue(instrument=instrument)
        thread = threading.Thread(target=self.build_history, args=[instrument, start, end, granularity_tuple])
        thread.daemon = True
        thread.start()

    def poll(self, instrument, interval=Interval(seconds=OandaGranularity.s5["sec"]),
             granularity=OandaGranularity.s5["api"]):
        while True:
            if self.stop_flag:
                break
            start = OandaDate(date=datetime.utcnow()).minus(seconds=interval.in_seconds())
            end = OandaDate(date=datetime.utcnow())
            print("Polling oanda for instrument: {} from {} to {}".format(instrument, start, end))
            response = Oanda.get_history_frame(instrument=instrument, start=start, end=end, granularity=granularity)
            self.frame_queues[instrument].enqueue(frame=response)
            sleep(interval.in_seconds())

    def build_history(self, instrument, start, end, granularity_tuple):
        rev = start.is_after(end)
        if rev:
            i = start
            start = end
            end = i
        while True:
            if self.stop_flag:
                break
            s, e = Oanda.range(start=start, end=end, interval=Interval(seconds=granularity_tuple["sec"]), rev=rev)
            response = Oanda.get_history_frame(instrument=instrument,
                                               start=s,
                                               end=e,
                                               granularity=granularity_tuple["api"])
            self.frame_queues[instrument].save(frame=response)
            if rev:
                end = s
            else:
                start = e
            sleep(cycle_duration)

    def stop(self):
        print("Stopping listener")
        self.stop_flag = True


if __name__ == "__main__":
    instruments = Reader.all_instruments()

    start = OandaDate().with_date(date_string="2018-01-15 18:11:50")
    end = OandaDate(date=datetime.utcnow()).minus(days=365)

    listener = Listener()
    for instrument in instruments:
        listener.start_building_history(instrument=instrument, start=start, end=end,
                                        granularity_tuple=OandaGranularity.s5)
        sleep(cycle_duration / len(instruments))

    while True:
        sleep(1000000)
