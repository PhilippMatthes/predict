from main.src.python.config import logdir_path
import os


def run_tensorboard():
    os.system("tensorboard --logdir {}".format(logdir_path))


if __name__ == "__main__":
    run_tensorboard()
