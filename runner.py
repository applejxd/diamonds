from modules import pre_process, validator, lgb_model
from hyperopt import fmin, hp, tpe


def main():
    cat_cols = ["cut", "color", "clarity"]
    process_ins = pre_process.PreProcess("./data/diamonds.csv", "price", cat_cols)
    train_x = process_ins.train_x
    train_y = process_ins.train_y

    model = lgb_model.LightGbmModel()
    validator_ins = validator.CrossValidator(train_x, train_y, 4)

    def eval_func(params):
        params.update({'objective': 'regression', 'metrics': 'mae'})
        score = validator_ins.validate(model, params)
        return score

    space = {
        'num_leaves': 50 + 10 * hp.randint('num_leaves', 16),
        'max_depth': 3 + hp.randint('max_depth', 8),
        'min_data_in_leaf': 5 + 2 * hp.randint('min_data_in_leaf', 11),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1.0),
        'learning_rate': hp.uniform('learning_rate', 0.03, 0.2),
        'subsample': hp.uniform('subsamplre', 0.5, 1.0)
    }
    best = fmin(eval_func, space=space,
                algo=tpe.suggest, max_evals=200)
    print(best)


if __name__ == "__main__":
    main()
