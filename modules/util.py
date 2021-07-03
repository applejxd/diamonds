import dask.dataframe as dd
import pandas as pd
from modules.self_logger import SelfLogger


class Util:
    _logger = None

    def __new__(cls):
        raise NotImplementedError("Cannot initialize via Constructor")

    @classmethod
    def _get_logger(cls):
        if cls._logger is None:
            cls._logger = SelfLogger.get_logger(__file__)
        return cls._logger

    @classmethod
    def read_csv(cls, file_name: str) -> pd.DataFrame:
        table = dd.read_csv(file_name).compute()
        cls._get_logger().info("CSV file read.")
        return table

    @classmethod
    def read_pickle(cls,file_name: str) -> pd.DataFrame:
        table = pd.read_pickle(file_name)
        cls._get_logger().info("Dataframe read from a pickle.")
        return table

    @classmethod
    def write_pickle(cls,df: pd.DataFrame, file_name: str) -> None:
        df.to_pickle(f"./result/{file_name}")
        cls._get_logger().info("Dataframe saved as a pickle.")
