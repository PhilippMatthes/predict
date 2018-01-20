import pathlib

import pandas as pd
from datetime import datetime


class FrameQueue:
    def __init__(self, instrument):
        self.instrument = instrument
        self.frames = [pd.DataFrame()]

    def __str__(self):
        return str(self.frames)

    def enqueue(self, frame):
        self.frames.append(frame)

    def save_all(self, path="./data"):
        full_path = "{}/{}".format(path, self.instrument)
        for frame in self.frames:
            file = pathlib.Path("{}/{}_to_{}.pickle".format(full_path,
                                                            frame.first_valid_index(),
                                                            frame.last_valid_index()))
            if not file.is_file():
                self.save(frame)

    def save(self, frame=pd.DataFrame(), path="./data"):
        if frame is None:
            return
        full_path = "{}/{}".format(path, self.instrument)
        pathlib.Path(full_path).mkdir(parents=True, exist_ok=True)
        frame.to_pickle("{}/{}_to_{}.pickle".format(full_path,
                                                    frame.first_valid_index(),
                                                    frame.last_valid_index()))

    def flush(self):
        self.frames.clear()

    def is_empty(self):
        return len(self.frames) == 0

    def next(self):
        return self.frames[0]
