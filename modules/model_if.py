import os
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from modules.util import Util
from modules.self_logger import SelfLogger


class ModelIF(ABC):
    def __init__(self):
        self._logger = SelfLogger.get_logger(__file__)
        path = f"../result/{__class__.__name__}.pkl"
        self._model = self.load_model() if os.path.exists(path) else None

    @abstractmethod
    def fit(self, tr_x: pd.DataFrame, tr_y: pd.Series,
            va_x: pd.DataFrame = None, va_y: pd.Series = None) -> None:
        pass

    @abstractmethod
    def predict(self, te_x: pd.DataFrame) -> np.ndarray:
        pass

    def save_model(self):
        Util.write_pickle(self._model, f"{__class__.__name__}.pkl")

    @staticmethod
    def load_model():
        return Util.read_pickle(f"{__class__.__name__}.pkl")
