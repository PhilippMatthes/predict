from datetime import datetime, timedelta, timezone


class OandaDate:
    def __init__(self, date=datetime.utcnow()):
        self.date = date

    def with_date(self, date_string):
        self.date = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
        return self

    def __str__(self):
        return self.date.isoformat("T") + "Z"

    def minus(self, weeks=0, days=0, hours=0, minutes=0, seconds=0):
        return OandaDate(self.date - timedelta(weeks=weeks, days=days, hours=hours, minutes=minutes, seconds=seconds))

    def plus(self, weeks=0, days=0, hours=0, minutes=0, seconds=0):
        return OandaDate(self.date + timedelta(weeks=weeks, days=days, hours=hours, minutes=minutes, seconds=seconds))

    def as_utc_datetime(self):
        return self.date

    def is_after(self, oanda_date):
        return oanda_date.date < self.date

    def is_before(self, oanda_date):
        return oanda_date.date > self.date

    def description(self):
        weekday = self.date.strftime("%A")
        day = self.date.strftime("%Y-%m-%d")
        time = self.date.strftime("%H:%M:%S")
        return "{} {} {}".format(weekday, day, time)
