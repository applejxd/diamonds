from modules import pre_process, validator, lgb_model
from hyperopt import fmin, hp, tpe
from modules.self_logger import SelfLogger


def runner():
    logger = SelfLogger.get_logger(__file__)
    logger.info("Use task runner.")

    cat_cols = ["cut", "color", "clarity"]
    process_ins = pre_process.PreProcess("./data/diamonds.csv", "price", cat_cols)
    train_x = process_ins.train_x
    train_y = process_ins.train_y

    model = lgb_model.LightGbmModel()
    model.tuning(train_x, train_y)


if __name__ == "__main__":
    runner()
