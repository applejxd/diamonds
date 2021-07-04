import os
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from modules.util import Util
from modules.self_logger import SelfLogger
from typing import Dict


class ModelIF(ABC):
    @property
    def space(self):
        return self._space

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, tmp):
        self._params.update(tmp)

    def __init__(self):
        self._logger = SelfLogger.get_logger(__file__)

        # ハイパーパラメータ
        self._space = None  # 探索空間
        self._params = None     # 学習に使うパラメータ

        # 機械学習モデル
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
