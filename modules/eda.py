import matplotlib.pyplot as plt
import util

# singleton
logger = util.make_logger(__name__)


def plot_carat(table):
    plt.scatter(table["carat"], table["price"], s=1)
    plt.xlabel("weight of the diamond [carat]")
    plt.ylabel("price [USD]")
    plt.show()


def main():
    table = util.read_csv("../data/diamonds.csv")
    plot_carat(table)


if __name__ == "__main__":
    main()
