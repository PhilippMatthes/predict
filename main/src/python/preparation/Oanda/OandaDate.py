from datetime import datetime, timedelta


class OandaDate:
    def __init__(self, date=datetime.now()):
        self.date = date

    def __str__(self):
        return self.date.strftime("%Y-%m-%d")

    def minus(self, weeks=0, days=0, minutes=0, seconds=0):
        return OandaDate(self.date - timedelta(weeks=weeks, days=days, minutes=minutes, seconds=seconds))

    def plus(self, weeks=0, days=0, minutes=0, seconds=0):
        return OandaDate(self.date + timedelta(weeks=weeks, days=days, minutes=minutes, seconds=seconds))
