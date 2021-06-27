from logging import Logger, getLogger, Formatter, StreamHandler, FileHandler, DEBUG
import dask.dataframe as dd
import pandas as pd


def make_logger(name: str) -> Logger:
    logger_ins = getLogger(name)
    logger_ins.setLevel(DEBUG)
    logger_ins.propagate = False

    formatter = Formatter("%(asctime)s: %(levelname)s: %(filename)s: %(message)s")

    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(formatter)

    file_handler = FileHandler(f"./logs/{name}.log")
    file_handler.setLevel(DEBUG)
    file_handler.setFormatter(formatter)

    logger_ins.addHandler(file_handler)
    logger_ins.addHandler(stream_handler)

    return logger_ins


logger = make_logger(__name__)


def read_csv(file_name: str) -> pd.DataFrame:
    table = dd.read_csv(file_name).compute()
    logger.info("CSV file has been read.")
    return table


def read_pickle(file_name: str) -> pd.DataFrame:
    table = pd.read_pickle(file_name)
    logger.info("Dataframe has been read from a pickle.")
    return table


def write_pickle(df: pd.DataFrame, file_name: str) -> None:
    df.to_pickle(f"./result/{file_name}")
    logger.info("Dataframe has been saved as a pickle.")
