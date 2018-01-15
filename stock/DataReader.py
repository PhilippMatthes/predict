import pandas_datareader.data as web
import datetime as dt
from Google

FRED_heap = web.DataReader([
    "DCOILWTICO",
    "VXOCLS",
    "DJIA",
    "DFF",
    "T10YIE",
    "DGS10",
    "DEXUSEU",
    "DEXCHUS",
    "DEXJPUS",
    "DEXUSUK",
    "GOLDAMGBD228NLBM",
    "BAMLH0A0HYM2",
    "USD3MTD156N",
    "DTB3",
    "SP500",
    "TEDRATE",
    "DTWEXM",
    "DCOILBRENTEU"
], "fred")

print(FRED_heap)