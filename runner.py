from modules import pre_process, validator, lgb_model


def main():
    cat_cols = ["cut", "color", "clarity"]
    process_ins = pre_process.PreProcess("./data/diamonds.csv", "price", cat_cols)
    train_x = process_ins.train_x
    train_y = process_ins.train_y

    model = lgb_model.LightGbmModel()
    params = {'objective': 'regression', 'metrics': 'mae',
              'seed': 71, 'verbose': 0}
    validator_ins = validator.CrossValidator(train_x, train_y, 4)
    validator_ins.validate(model, params)


if __name__ == "__main__":
    main()
