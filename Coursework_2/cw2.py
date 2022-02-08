import matplotlib.pyplot as plt
import numpy as np


def sorted_barplot(P1, P2, W):
    """
    Function for making a sorted bar plot based on values in P, and labelling the plot with the
    corresponding names
    :param P: An array of length num_players (107)
    :param W: Array containing names of each player
    :return: None
    """
    M = len(P1)
    width = 0.1
    xx = np.linspace(0, M, M)
    fig=plt.figure(tight_layout=True)
    fig.set_size_inches(6.27, 11.69, forward=True)
    sorted_indices1 = np.argsort(P1)
    sorted_names1 = W[sorted_indices1]
    plt.barh(xx, P1[sorted_indices1], label="Message Passing", width=width)
    plt.yticks(np.linspace(0, M, M) + width/2, labels=sorted_names1[:, 0], fontsize=6)
    plt.barh(xx, P2[sorted_indices1], label="Gibbs", width=width)
    plt.yticks(np.linspace(0, M, M), labels=sorted_names1[:, 0], fontsize=6)
    plt.ylim([-2, 109])
    plt.xlabel("Average Probability of Winning", fontsize=16)
    plt.ylabel("Players Ranking", fontsize=16)
    plt.title("Player Ranking Predictions", fontsize=16)
    plt.savefig('Figure8_Difference.png')
    plt.show()
