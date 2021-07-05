from abc import ABC
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
        self._params = {'objective': 'regression_l1', 'metrics': 'mae',
                        'force_col_wise': 'true', 'learning_rate': 0.1,
                        'num_leaves': 19}
        self._space = {
            # 'num_leaves': 10 + 10 * hp.randint('num_leaves', 37),
            'min_data_in_leaf': 5 + 2 * hp.randint('min_data_in_leaf', 11),
            # 'max_depth': 3 + hp.randint('max_depth', 6),
        }
        self.importance = None

    def fit(self, tr_x: pd.DataFrame, tr_y: pd.Series,
            va_x: pd.DataFrame = None, va_y: pd.Series = None):
        # 特徴量と目的変数を lightgbm のデータ構造に変換する
        lgb_train = lgb.Dataset(tr_x, tr_y)
        lgb_eval = lgb.Dataset(va_x, va_y)

        # early stopping を利用 (num_boost_round, early_stopping_rounds)
        # バリデーションデータもモデルに渡し、スコアがどう変わるかモニタリング
        try:
            self._model = lgb.train(self._params, lgb_train, verbose_eval=False,
                                    valid_names=['train', 'valid'], valid_sets=[lgb_train, lgb_eval],
                                    num_boost_round=10000, early_stopping_rounds=50)
        except Exception as e:
            self._logger.error(f"Error occurred at learning: {e}")

        # 特徴量の重要度表示
        importance = pd.DataFrame(self._model.feature_importance(), index=tr_x.columns,
                                  columns=["importance"])
        self.importance = importance

    def predict(self, te_x: pd.DataFrame) -> np.ndarray:
        pred = self._model.predict(te_x)
        return pred

    def tuning(self, train_x, train_y):
        validator_ins = validator.CrossValidator(train_x, train_y, 4)

        def eval_func(params):
            self.params = params
            score = validator_ins.validate(self)
            return score

        best = fmin(eval_func, space=self.space,
                    algo=tpe.suggest, max_evals=200)
        self.save_model()
        self._logger.debug(best)
        self._logger.debug(self.importance)
