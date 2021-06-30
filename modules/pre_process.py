import pandas as pd
from sklearn.preprocessing import LabelEncoder
from modules.self_logger import SelfLogger
import util
from typing import List
import os


class PreProcess:
    def __init__(self):
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        self.logger = SelfLogger.get_inst().get_logger(file_name)

    def categorical_encoder(self, table: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        """
        カテゴリ変数をループしてlabel encoding
        :return:
        """
        result = table.copy()
        for category in cat_cols:
            # 学習データに基づいて定義する
            encoder = LabelEncoder()
            encoder.fit(result[category])
            result[category] = encoder.transform(result[category])
        self.logger.info("Categorical variables converted.")
        return result

    def pre_process(self):
        cat_cols = ["cut", "color", "clarity"]
        table = util.read_csv("../data/diamonds.csv")
        table = self.categorical_encoder(table, cat_cols)
        print(table)


if __name__ == "__main__":
    PreProcess().pre_process()
