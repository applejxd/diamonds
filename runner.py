from modules import pre_process, validator, lgb_model
from hyperopt import fmin, hp, tpe
from modules.self_logger import SelfLogger


def runner():
    SelfLogger.get_logger(__file__).info("Use task runner.")

    cat_cols = ["cut", "color", "clarity"]
    process_ins = pre_process.PreProcess("./data/diamonds.csv", "price", cat_cols)
    train_x = process_ins.train_x
    train_y = process_ins.train_y

    model = lgb_model.LightGbmModel()
    validator_ins = validator.CrossValidator(train_x, train_y, 4)

    def eval_func(params):
        model.params = params
        score = validator_ins.validate(model)
        return score

    best = fmin(eval_func, space=model.space,
                algo=tpe.suggest, max_evals=200)
    print(best)


if __name__ == "__main__":
    runner()
