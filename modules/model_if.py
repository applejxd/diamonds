import json
import os
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, space_eval

from modules import validator
from modules.self_logger import SelfLogger
from typing import Dict
import pickle


def convert_dict_type(my_dict):
    new_dict = {}
    for key, value in my_dict.items():
        if type(value) is np.int64:
            new_dict[key] = int(value)
        else:
            new_dict[key] = value
    return new_dict


class ModelIF(ABC):
    def __init__(self):
        self._logger = SelfLogger.get_logger(__file__)

        # ハイパーパラメータ
        self._params: Dict = {}
        if os.path.exists(f"../output/{type(self).__name__}.json"):
            with open(f"../output/{type(self).__name__}.json") as f:
                self._params = json.load(f)

        # ハイパーパラメータの探索空間
        # cf. https://hyperopt.github.io/hyperopt/getting-started/search_spaces/#parameter-expressions
        self._space: Dict = {}

        # 機械学習モデル
        path = f"../output/{type(self).__name__}.pkl"
        self._model = self.load_model() if os.path.exists(path) else None

    @abstractmethod
    def fit(self, tr_x: pd.DataFrame, tr_y: pd.Series,
            va_x: pd.DataFrame = None, va_y: pd.Series = None):
        pass

    @abstractmethod
    def predict(self, te_x: pd.DataFrame) -> np.ndarray:
        pass

    def tuning(self, train_x, train_y):
        validator_ins = validator.CrossValidator(train_x, train_y, 4)

        def eval_func(params):
            self._params = convert_dict_type(params)
            score = validator_ins.validate(self)
            return score

        best_params: Dict = fmin(eval_func, space=self._space,
                                 algo=tpe.suggest, max_evals=200)
        self._params.update(convert_dict_type(space_eval(self._space, best_params)))

        with open(f"./output/{type(self).__name__}.json", "w") as f:
            json.dump(self._params, f, indent=2)

        self._logger.debug(best_params)

    def save_model(self):
        with open(f"./output/{type(self).__name__}.pkl", "wb") as f:
            pickle.dump(self._model, f)

    def load_model(self):
        with open(f"./output/{type(self).__name__}.pkl", "rb") as f:
            result = pickle.load(f)
        return result
