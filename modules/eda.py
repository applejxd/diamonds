import matplotlib.pyplot as plt
import pandas as pd
from modules.self_logger import SelfLogger
from util import Util

logger = SelfLogger.get_logger(__file__)


def plot_carat(table: pd.DataFrame):
    plt.scatter(table["carat"], table["price"], s=1)
    plt.xlabel("weight of the diamond [carat]")
    plt.ylabel("price [USD]")
    plt.show()


def eda():
    table = Util.read_csv("../data/diamonds.csv")
    plot_carat(table)


if __name__ == "__main__":
    eda()
