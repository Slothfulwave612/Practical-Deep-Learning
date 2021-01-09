"""
__author__: Anmol_Durgapal(@slothfulwave612)

Python module containing utility functions.
"""
# required packages/modules
import matplotlib.pyplot as plt
from matplotlib import rcParams


def plot_over_epochs(
    train_list, valid_list, title=None, ylabel=None, path=None
):
    """
    Function to plot a line plot for train and valid stats.

    Args:
        train_list (list): containing training stats.
        valid_list (list): containing validation stats.
        title (str, optional): title of the plot. Defaults to None.
        ylabel (str, optional): ylabel for the plot. Defaults to None.
        path (str, optional): path where plot will be saved. Defaults to None.

    Returns:
        figure.Figure: figure object.
        axes.Axes: axes object.
    """
    # default font-family
    rcParams["font.family"] = "serif"

    # create subplot
    fig, ax = plt.subplots(facecolor="#F2F2F2", figsize=(12, 8))
    ax.set_facecolor("#F2F2F2")

    # plot train stats
    ax.plot(
        range(len(train_list)), train_list,
        color="crimson", ls="--", label="Train"
    )
    ax.plot(
        range(len(valid_list)), valid_list,
        color="#222222", ls=":", label="Valid"
    )

    # set title and labels
    ax.set_title(title, size=20, color="#121212")
    ax.set_xlabel("Epochs", size=14, color="#121212")
    ax.set_ylabel(ylabel, size=14, color="#121212")

    # legend for the plot
    ax.legend(loc=0)

    # grid
    ax.grid()

    if path:
        fig.savefig(path, dpi=600, bbox_inches="tight")

    return fig, ax
