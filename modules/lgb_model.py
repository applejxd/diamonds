from abc import ABC
from modules.model_if import ModelIF
import lightgbm as lgb
import pandas as pd
import numpy as np


class LightGbmModel(ModelIF, ABC):
    def __init__(self):
        super().__init__()

    def fit(self, tr_x: pd.DataFrame, tr_y: pd.Series,
            va_x: pd.DataFrame = None, va_y: pd.Series = None) -> None:
        # 特徴量と目的変数を lightgbm のデータ構造に変換する
        lgb_train = lgb.Dataset(tr_x, tr_y)
        lgb_eval = lgb.Dataset(va_x, va_y)

        # ハイパーパラメータの設定
        # 回帰タスク, MAE を指標に最適化. 再現性のために seed を指定.
        params = {'objective': 'regression', 'metrics': 'mae',
                  'seed': 71, 'verbose': 0}
        self._logger.debug(f"param = {params}")

        # 学習の実行
        # カテゴリ変数をパラメータで指定
        # バリデーションデータもモデルに渡し、スコアがどう変わるかモニタリング
        num_round = 100
        self._logger.debug(f"num_round = {num_round}")
        try:
            self._model = lgb.train(params, lgb_train, num_boost_round=num_round,
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
