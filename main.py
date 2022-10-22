from multiprocessing import Pool
from config import DefaultConfig
from model.fdkt import FDKT


def run(arg):
    h = FDKT(arg)
    h.train_and_test(arg)


if __name__ == '__main__':
    config = DefaultConfig()
    run(["bridge_algebra06", "fold0"])
