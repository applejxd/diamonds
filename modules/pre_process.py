from sklearn.preprocessing import LabelEncoder
import util


logger = util.make_logger(__name__)


def categorical_encoder(table, cat_cols):
    """
    カテゴリ変数をループしてlabel encoding
    :return:
    """
    for category in cat_cols:
        # 学習データに基づいて定義する
        encoder = LabelEncoder()
        encoder.fit(table[category])
        table[category] = encoder.transform(table[category])
    logger.info("Categorical variables have been converted.")
    return table


def main():
    cat_cols = ["cut", "color", "clarity"]
    table = util.read_csv("../data/diamonds.csv")
    table = categorical_encoder(table, cat_cols)
    print(table)


if __name__ == "__main__":
    main()
