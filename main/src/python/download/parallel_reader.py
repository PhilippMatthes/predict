import glob
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

from main.src.python.config import data_path
from main.src.python.config import config_path
from main.src.python.download.index_file import IndexFile


class ParallelReader:
    def __init__(self, start, end, read=True, reduce=False):
        self.df_dict = {}
        self.stacked_df_dict = {}
        self.adjusted_df_dict = {}
        self.data = None
        if read:
            self.read(start, end)
            self.stack()
            self.adjust()
            if reduce:
                self.reduce()
            self.concat()

    @staticmethod
    def all_instruments():
        with open(os.path.join(config_path, "instruments.txt")) as f:
            content = f.readlines()
        return [x.strip() for x in content]

    def read(self, start, end):
        print("Reading data frames")
        instruments = ParallelReader.all_instruments()
        for instrument in instruments:
            paths = glob.glob("{}/{}/*.pickle".format(data_path, instrument))
            for path in paths:
                file = IndexFile(path)
                if file.needed_between(start, end):
                    self.df_dict.setdefault(instrument, []).append(file.read_between(start, end))

    def stack(self):
        print("Stacking read data frames")
        for instrument in self.df_dict.keys():
            frames = self.df_dict[instrument]
            concatenated_frames = pd.concat(frames)
            removed_dupes = concatenated_frames[~concatenated_frames.index.duplicated(keep="first")]
            self.stacked_df_dict[instrument] = removed_dupes

    def adjust(self):
        print("Adjusting stacked data frames")
        frames = [x for x in self.stacked_df_dict.values()]
        concatenated_frames = pd.concat(frames)
        removed_dupes = concatenated_frames[~concatenated_frames.index.duplicated(keep="first")]
        sorted_frames = removed_dupes.sort_index()
        index = sorted_frames.index
        for instrument, stacked_df in self.stacked_df_dict.items():
            self.adjusted_df_dict[instrument] = stacked_df.reindex(index=index, method="nearest")

    def reduce(self):
        print("Reducing adjusted data frames")
        for instrument, adjusted_df in self.adjusted_df_dict.items():
            self.adjusted_df_dict[instrument] = adjusted_df[adjusted_df.columns[1]]

    def concat(self):
        print("Concatenating reduced data frames")
        df = pd.concat([self.adjusted_df_dict[key] for key in sorted(self.adjusted_df_dict.keys())], axis=1)
        print("Converting concatenated data frames to numpy array")
        self.data = df.values


if __name__ == "__main__":
    reader = Reader(read=True)
