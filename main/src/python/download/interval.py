
class Interval:
    def __init__(self, weeks=0, days=0, minutes=0, seconds=0):
        self.weeks = weeks
        self.days = days
        self.minutes = minutes
        self.seconds = seconds

    def in_seconds(self):
        return self.seconds + (60*self.minutes) + (86400*self.days) + (604800*self.weeks)
