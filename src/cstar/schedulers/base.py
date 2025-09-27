from abc import ABC, abstractmethod

import numpy as np


class Scheduler(ABC):
    @abstractmethod
    def allocate(self, demand: np.ndarray, capacity: np.ndarray, B: float, ctx):
        raise NotImplementedError
