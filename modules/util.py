import dask.dataframe as dd
import pandas as pd
from modules.self_logger import SelfLogger

logger = None


def get_logger():
    global logger
    if not logger:
        logger = SelfLogger.get_logger(__file__)
    return logger


def read_csv(file_name: str) -> pd.DataFrame:
    table = dd.read_csv(file_name).compute()
    get_logger().info("CSV file read.")
    return table


def read_pickle(file_name: str) -> pd.DataFrame:
    table = pd.read_pickle(file_name)
    get_logger().info("Dataframe read from a pickle.")
    return table


def write_pickle(df: pd.DataFrame, file_name: str) -> None:
    df.to_pickle(f"./result/{file_name}")
    get_logger().info("Dataframe saved as a pickle.")
