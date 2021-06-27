import pandas as pd
from sklearn.preprocessing import LabelEncoder
import util
from typing import List

logger = util.make_logger(__name__)


def categorical_encoder(table: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
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
    logger.info("Categorical variables have been converted.")
    return result


def pre_process():
    cat_cols = ["cut", "color", "clarity"]
    table = util.read_csv("../data/diamonds.csv")
    table = categorical_encoder(table, cat_cols)
    print(table)


if __name__ == "__main__":
    pre_process()
