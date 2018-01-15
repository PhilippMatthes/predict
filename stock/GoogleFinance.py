import csv
import datetime
import re
import urllib.request
from urllib.error import HTTPError

import pandas as pd


class GoogleFinance:

    @staticmethod
    def get_finance_intraday(ticker, period=60, days=1):
        """
        Retrieve intraday stock data from Google Finance.

        Parameters
        ----------
        ticker : str
            Company ticker symbol.
        period : int
            Interval between stock values in seconds.
        days : int
            Number of days of data to retrieve.

        Returns
        -------
        df : pandas.DataFrame
            DataFrame containing the opening price, high price, low price,
            closing price, and volume. The index contains the times associated with
            the retrieved price values.
        """

        url = 'http://www.google.com/finance/getprices?i={period}&p={days}d&f=d,o,h,l,c,v&df=cpct&q={ticker}' \
            .format(ticker=ticker, period=period, days=days)
        print(url)
        try:
            page = urllib.request.urlopen(url)
            csvfile = page.read().decode('utf-8')
            reader = csv.reader(csvfile.splitlines())
            columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            rows = []
            times = []
            for row in reader:
                if re.match('^[a\d]', row[0]):
                    if row[0].startswith('a'):
                        start = datetime.datetime.fromtimestamp(int(row[0][1:]))
                        times.append(start)
                    else:
                        times.append(start + datetime.timedelta(seconds=period * int(row[0])))
                    rows.append(map(float, row[1:]))
            if len(rows):
                return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name='Date'),
                                    columns=columns)
            else:
                return pd.DataFrame(rows, index=pd.DatetimeIndex(times, name='Date'))
        except HTTPError:
            print("Warning! Google refused to respond!")
            return None


if __name__ == "__main__":
    print(GoogleFinance.get_finance_intraday("AAPL", period=60, days=1))
