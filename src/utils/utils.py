import os
import random
import time
from contextlib import contextmanager

import numpy as np


def seed_everything(seed: int):
    """乱数シードを設定"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


@contextmanager
def timer(name: str):
    """時間計測用のcontextmanager"""
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")
