import json
import os
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import hyperopt as hopt

from modules import validator
from modules.self_logger import SelfLogger
from typing import Dict
from datetime import date, datetime
import pickle


def json_serial(obj):
    """
    オリジナルの json シリアライザ
    cf. https://www.yoheim.net/blog.php?q=20170703

    :param obj: json の値
    :return:    シリアライズの結果
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, np.int64):
        return obj.item()
    # 上記以外はサポート対象外.
    raise TypeError("Type %s not serializable" % type(obj))


class ModelIF(ABC):
    def __init__(self):
        self._logger = SelfLogger.get_logger(__file__)

        # ハイパーパラメータ
        self._params: Dict = {}
        if os.path.exists(f"../output/{type(self).__name__}_params.json"):
            with open(f"../output/{type(self).__name__}_params.json") as f:
                self._params = json.load(f)

        # ハイパーパラメータの探索空間
        # cf. https://hyperopt.github.io/hyperopt/getting-started/search_spaces/#parameter-expressions
        self._space: Dict = {}

        # 機械学習モデル
        self._model = None
        path = f"../output/{type(self).__name__}.pkl"
        if os.path.exists(path):
            self.load_model()

    @abstractmethod
    def fit(self, tr_x: pd.DataFrame, tr_y: pd.Series,
            va_x: pd.DataFrame = None, va_y: pd.Series = None):
        pass

    @abstractmethod
    def predict(self, te_x: pd.DataFrame) -> np.ndarray:
        pass

    def tuning(self, validator_ins):
        """
        hyperopt を使用したハイパーパラメータチューニング
        cf. http://hyperopt.github.io/hyperopt/

        :param validator_ins: バリデーションデータを取得できるインスタンス
        """
        def eval_func(params):
            self._params = params
            score = validator_ins.validate(self)
            return {"loss": score, "status": hopt.STATUS_OK}

        trials = hopt.Trials()
        best_params: Dict = hopt.fmin(eval_func, space=self._space,
                                      algo=hopt.tpe.suggest, max_evals=100,
                                      trials=trials)
        self._params.update(hopt.space_eval(self._space, best_params))

        # 最良のパラメータを保存
        with open(f"./output/{type(self).__name__}_params.json", "w") as f:
            json.dump(self._params, f, indent=2, default=json_serial)
        # 最適化のステータスを保存
        with open(f"./output/{type(self).__name__}_status.json", "w") as f:
            json.dump(trials.best_trial, f, indent=2, default=json_serial)

        self._logger.debug(f'best loss = {trials.best_trial["result"]["loss"]}')
        self._logger.debug(best_params)

    def save_model(self):
        with open(f"./output/{type(self).__name__}.pkl", "wb") as f:
            pickle.dump(self._model, f)

    def load_model(self):
        with open(f"./output/{type(self).__name__}.pkl", "rb") as f:
            self._model = pickle.load(f)
