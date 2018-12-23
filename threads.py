import threading
from connected_components import *
from fractal import *
from basic_features import *
from enclosed_regions import enclosed_regions


class Thread_f1(threading.Thread):
    def __init__(self, line):
        threading.Thread.__init__(self)
        self.line = line
        self.features = []

    def run(self):
        self.features = line_features(self.line)


class Thread_f2(threading.Thread):
    def __init__(self, line):
        threading.Thread.__init__(self)
        self.line = line
        self.features = []

    def run(self):
        self.features = basic_ftrs(self.line)


class Thread_f3(threading.Thread):
    def __init__(self, line):
        threading.Thread.__init__(self)
        self.line = line
        self.features = []

    def run(self):
        self.features = getfractalftrs(self.line)


class Thread_f4(threading.Thread):
    def __init__(self, line):
        threading.Thread.__init__(self)
        self.line = line
        self.features = []

    def run(self):
        self.features = enclosed_regions(self.line)

