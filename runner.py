from modules import pre_process, validator, lgb_model


def main():
    cat_cols = ["cut", "color", "clarity"]
    process_ins = pre_process.PreProcess("./data/diamonds.csv", "price", cat_cols)
    train_x = process_ins.train_x
    train_y = process_ins.train_y

    model = lgb_model.LightGbmModel()
    validator_ins = validator.CrossValidator(4)
    validator_ins.validate(train_x, train_y, model)


if __name__ == "__main__":
    main()
