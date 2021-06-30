import dask.dataframe as dd
import pandas as pd
from modules.self_logger import SelfLogger


def read_csv(file_name: str) -> pd.DataFrame:
    table = dd.read_csv(file_name).compute()
    SelfLogger.get_inst().get_logger(__file__).info("CSV file  read.")
    return table


def read_pickle(file_name: str) -> pd.DataFrame:
    table = pd.read_pickle(file_name)
    SelfLogger.get_inst().get_logger(__file__).info("Dataframe read from a pickle.")
    return table


def write_pickle(df: pd.DataFrame, file_name: str) -> None:
    df.to_pickle(f"./result/{file_name}")
    SelfLogger.get_inst().get_logger(__file__).info("Dataframe saved as a pickle.")
