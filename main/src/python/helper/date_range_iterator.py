
class DateRangeIterator:
    def __init__(self, start, end, interval):
        if start.is_after(end):
            raise Exception("Start has to be before end!")
        self.start = start
        self.end = end
        self.interval = interval

    def has_next(self):
        return self.start.plus(seconds=self.interval.in_seconds()).is_before(self.end)

    def next(self):
        start = self.start
        self.start = self.start.plus(seconds=self.interval.in_seconds())
        return start, self.start
