import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from modules.self_logger import SelfLogger


class CrossValidator:
    def __init__(self, train_x: pd.DataFrame, train_y: pd.Series, n_fold=4):
        self._logger = SelfLogger.get_logger(__file__)
        self._logger.info("Execute cross validation.")

        # 分割数
        self._n_fold = n_fold
        self._logger.debug(f"split number = {n_fold}")

        self._tr_x_list, self._tr_y_list = [], []
        self._va_x_list, self._va_y_list = [], []
        self._get_fold(train_x, train_y)

    def _get_fold(self, train_x: pd.DataFrame, train_y: pd.Series):
        # KFoldクラスを用いてクロスバリデーションの分割を行う
        seed = 42
        kf = KFold(n_splits=self._n_fold, shuffle=True, random_state=seed)
        self._logger.debug(f"Seed of KFold = {seed}")

        for tr_idx, va_idx in kf.split(train_x):
            tr_x, va_x = train_x.iloc[tr_idx], train_x.iloc[va_idx]
            tr_y, va_y = train_y.iloc[tr_idx], train_y.iloc[va_idx]

            self._tr_x_list.append(tr_x)
            self._tr_y_list.append(tr_y)
            self._va_x_list.append(va_x)
            self._va_y_list.append(va_y)

    def validate(self, model_ins):
        """
        クロスバリデーション

        :param model_ins: 機械学習モデル（ダックタイピング）
        :return: 評価値
        """
        scores = []
        for tr_x, tr_y, va_x, va_y \
                in zip(self._tr_x_list, self._tr_y_list, self._va_x_list, self._va_y_list):
            # 学習の実行、バリデーションデータの予測値の出力、スコアの計算を行う
            model_ins.fit(tr_x, tr_y, va_x, va_y)
            va_pred = model_ins.predict(va_x)
            score = mean_absolute_error(va_y, va_pred)
            scores.append(score)

        # 各foldのスコアの平均をとる
        result = np.mean(scores)
        # self._logger.debug(f"MAE of the cross validation = {result}")
        return result
