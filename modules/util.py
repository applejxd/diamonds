import dask.dataframe as dd
import pandas as pd
from modules.self_logger import SelfLogger
import pickle


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
    def read_pickle(cls, file_name: str) -> pd.DataFrame:
        with open(f"./result/{file_name}.pkl", "rb") as f:
            result = pickle.load(f)
        cls._get_logger().info("Data read from a pickle.")
        return result

    @classmethod
    def write_pickle(cls, data, file_name: str) -> None:
        with open(f"./result/{file_name}.pkl", "wb") as f:
            pickle.dump(data, f)
        cls._get_logger().info("Data saved as a pickle.")
