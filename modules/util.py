import dask.dataframe as dd
import pandas as pd
from modules.self_logger import SelfLogger

logger = SelfLogger.make_logger(__name__)


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
