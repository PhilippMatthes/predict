from main.src.python.config import sessions_path
import os


def run_tensorboard():
    os.system("tensorboard --logdir {}".format(sessions_path))


if __name__ == "__main__":
    run_tensorboard()
