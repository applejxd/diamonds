from abc import ABC
from typing import Dict

from modules.model_if import ModelIF
import lightgbm as lgb
import pandas as pd
import numpy as np
from modules import validator
from hyperopt import fmin, hp, tpe


class LightGbmModel(ModelIF, ABC):
    def __init__(self):
        super().__init__()
        # 学習タスクの定義・パラメータ調整用に学習率は高く設定
        self._space = {
            'objective': 'regression_l1', 'metrics': 'mae',
            'force_col_wise': 'true', 'learning_rate': 0.1,
            'num_leaves': 19,
            # 'num_leaves': 10 + 10 * hp.randint('num_leaves', 37)
            'min_data_in_leaf': 9,
            # 'min_data_in_leaf': 5 + 2 * hp.randint('min_data_in_leaf', 11),
            'max_depth': 3 + hp.randint('max_depth', 6)
        }
        self._importance = pd.DataFrame(index=[])

    def fit(self, tr_x: pd.DataFrame, tr_y: pd.Series,
            va_x: pd.DataFrame = None, va_y: pd.Series = None):
        # 特徴量と目的変数を lightgbm のデータ構造に変換する
        lgb_train = lgb.Dataset(tr_x, tr_y)
        lgb_eval = lgb.Dataset(va_x, va_y)

        # early stopping を利用 (num_boost_round, early_stopping_rounds)
        # バリデーションデータもモデルに渡し、スコアがどう変わるかモニタリング
        try:
            self._model = lgb.train(self._params, lgb_train, num_boost_round=100,
                                    valid_sets=[lgb_train, lgb_eval], valid_names=['train', 'valid'],
                                    callbacks=[lgb.log_evaluation(10), lgb.early_stopping(20)])
        except Exception as e:
            self._logger.error(f"Error occurred at learning: {e}")

        # 特徴量の重要度表示
        importance = pd.DataFrame(self._model.feature_importance(), index=tr_x.columns,
                                  columns=["importance"])
        self._importance = importance

    def predict(self, te_x: pd.DataFrame) -> np.ndarray:
        pred = self._model.predict(te_x)
        return pred

    def tuning(self, validator_ins):
        super().tuning(validator_ins)
        self._logger.debug(self._importance)
