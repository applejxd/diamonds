import pandas as pd
from sklearn.preprocessing import LabelEncoder
from modules.self_logger import SelfLogger
from util import Util
from typing import List


class PreProcess:
    def __init__(self, file_path, target_col, cat_cols):
        self._logger = SelfLogger.get_logger(__file__)

        # 読み込み
        table = Util.read_csv(file_path)
        self._train_x = table.drop(target_col, axis=1)
        self._train_y = table[target_col]

        # 前処理
        self._categorical_encoder(cat_cols)

    @property
    def train_x(self):
        return self._train_x

    @property
    def train_y(self):
        return self._train_y

    def _categorical_encoder(self, cat_cols: List[str]) -> None:
        """
        カテゴリ変数をループしてlabel encoding
        :return:
        """
        for category in cat_cols:
            # 学習データに基づいて定義する
            encoder = LabelEncoder()
            encoder.fit(self._train_x[category])
            self._train_x[category] = encoder.transform(self._train_x[category])
        self._logger.info("Categorical variables converted.")


if __name__ == "__main__":
    PreProcess("../data/diamonds.csv", "price", ["cut", "color", "clarity"])
