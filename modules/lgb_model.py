from abc import ABC
from modules.model_if import ModelIF
import lightgbm as lgb
import pandas as pd
import numpy as np
from typing import Dict
from hyperopt import hp


class LightGbmModel(ModelIF, ABC):
    def __init__(self):
        super().__init__()
        self._params = {'objective': 'regression', 'metrics': 'mae'}
        self._space = {
            'num_leaves': 50 + 10 * hp.randint('num_leaves', 16),
            'min_data_in_leaf': 5 + 2 * hp.randint('min_data_in_leaf', 11),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
            'learning_rate': hp.uniform('learning_rate', 0.03, 0.2),
            'subsample': hp.uniform('subsamplre', 0.5, 1.0)
        }

    def fit(self, tr_x: pd.DataFrame, tr_y: pd.Series,
            va_x: pd.DataFrame = None, va_y: pd.Series = None) -> None:
        # 特徴量と目的変数を lightgbm のデータ構造に変換する
        lgb_train = lgb.Dataset(tr_x, tr_y)
        lgb_eval = lgb.Dataset(va_x, va_y)

        # 学習の実行
        # カテゴリ変数をパラメータで指定
        # バリデーションデータもモデルに渡し、スコアがどう変わるかモニタリング
        num_round = 100
        self._logger.debug(f"num_round = {num_round}")
        try:
            self._model = lgb.train(self._params, lgb_train, num_boost_round=num_round,
                                    valid_names=['train', 'valid'],
                                    valid_sets=[lgb_train, lgb_eval])
        except Exception as e:
            self._logger.error(f"Error occurred at learning: {e}")

        # 特徴量の重要度表示
        importance = pd.DataFrame(self._model.feature_importance(), index=tr_x.columns,
                                  columns=["importance"])
        self._logger.debug(f"feature importance = {importance}")

        self.save_model()

    def predict(self, te_x: pd.DataFrame) -> np.ndarray:
        pred = self._model.predict(te_x)
        return pred
