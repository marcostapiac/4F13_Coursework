import scipy
import pandas as pd
from gibbsrank import gibbs_sample
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from gibbsrank import gibbs_sample
from eprank import eprank
import pandas
from emcee import autocorr
from cw2 import sorted_barplot


def load_data(w=0):
    # set seed for reproducibility
    np.random.seed(0)
    # load data
    data = sio.loadmat('tennis_data.mat')
    # Array containing the names of each player
    W = data['W']
    # loop over array to format more nicely
    for i, player in enumerate(W):
        W[i] = player[0]
    # Array of size num_games x 2. The first entry in each row is the winner of game i, the second is the loser
    G = data['G'] - 1
    # Number of players
    M = W.shape[0]
    # Number of Games
    N = G.shape[0]
    if w == 1:
        return G, M, W
    return G, M


def ACF(skill_samples):
    # Code for plotting the autocorrelation function for player p and computing autocorrelation time
    p = 0
    autocor = np.zeros(10)
    for i in range(10):
        autocor[i] = pandas.Series.autocorr(pandas.Series(skill_samples[p, :]), lag=i)
    plt.plot(autocor)
    return autocorr.integrated_time(skill_samples[p, :])


def MessagePassing(G, numPlayers, num_iters, w=0):
    p = 1
    pv = 0.5
    return eprank(G, numPlayers, num_iters, p, pv, w)


def MessagePassing1(G, numPlayers, num_iters, p):
    pv = 0.5
    # run message passing algorithm, returns mean and precision for each player    pv = 0.5
    mean_player_skills, precision_player_skills, mean, precisions = eprank(G, numPlayers, num_iters, p, pv)
    diffs_precisions = np.array([np.abs(precisions[i + 1] - precisions[i]) for i in range(num_iters - 1)])
    diffs_mean = np.array([np.abs(mean[i + 1] - mean[i]) for i in range(num_iters - 1)])

    return diffs_mean, diffs_precisions


def MessagePassing2(G, numPlayers, num_iters, p=1):
    # run message passing algorithm, returns mean and precision for each player
    pv = [0.05, 0.5, 1, 1.5]
    DMs = np.zeros((4, num_iters - 1))

    for i in range(len(pv)):
        mean_player_skills, precision_player_skills, mean, precisions = eprank(G, numPlayers, num_iters, p, pv[i])
        diffs_mean = np.array([np.abs(mean[i + 1] - mean[i]) for i in range(num_iters - 1)])
        DMs[i] = diffs_mean
    return DMs


def Ex1():
    G, M = load_data()
    I = 1100
    skill_samples = gibbs_sample(G, M, I)
    iters = np.linspace(1, I, I)
    plt.subplot(3, 1, 1)
    plt.plot(iters, skill_samples[1 - 1, :], 'b', label="Rafael Nadal Sampled Skills")
    plt.xlabel("Gibbs Sampler Iterations")
    plt.ylabel("Sampled Player Skills")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(iters, skill_samples[68 - 1, :], 'r', label="Igor Kunitsyn Sampled Skills")
    plt.xlabel("Gibbs Sampler Iterations")
    plt.ylabel("Sampled Player Skills")
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot(iters, skill_samples[102 - 1, :], 'g', label="Rohan-Bopanna Sampled Skills")
    plt.xlabel("Gibbs Sampler Iterations")
    plt.ylabel("Sampled Player Skills")
    plt.legend()
    plt.suptitle("Player Skills From Gibbs Sampler")
    plt.show()


def Ex1_Burn_in_estimation():
    G, M = load_data()
    I = 200
    skill_samples = gibbs_sample(G, M, I)
    iters = np.linspace(1, I, I)
    plt.subplot(1, 2, 1)
    plt.plot(iters, skill_samples[1 - 1, :], 'b', label="Rafael Nadal Sampled Skills")
    plt.xlabel("Gibbs Sampler Iterations", fontsize="12")
    plt.xticks(np.arange(min(iters) - 1, max(iters) + 1, 20))
    plt.ylabel("Sampled Player Skills", fontsize="12")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(iters, skill_samples[68 - 1, :], 'r', label="Igor Kunitsyn Sampled Skills")
    plt.xlabel("Gibbs Sampler Iterations", fontsize="12")
    plt.xticks(np.arange(min(iters) - 1, max(iters) + 1, 20))
    plt.ylabel("Sampled Player Skills", fontsize="12")
    plt.rc('legend', fontsize=12)  # using a size in points
    plt.legend()
    plt.suptitle('Burn-In Estimation for Players 1 and 68')
    plt.show()


def Ex1_Autocorr():
    G, M = load_data()
    I = 11
    skill_samples = gibbs_sample(G, M, I)
    iters = np.linspace(1, I, I)
    autocorr_time = ACF(skill_samples)
    print(autocorr_time)
    # plt.plot(iters, skill_samples[1, :])
    plt.show()


def Ex2():
    G, M = load_data()
    I = 50
    diffs_mean, diffs_precisions = MessagePassing1(G, M, I, 68)
    threshold = 1e-3
    min_error = np.ones(I - 1) * threshold
    linspace = np.linspace(1, I, num=I - 1)
    plt.plot(linspace[5:], diffs_mean[5:], 'b', label="Mean for Player 68")
    plt.plot(linspace[5:], diffs_precisions[5:], 'g', label="Precision for Player 68")
    plt.plot(linspace[5:], min_error[5:], 'r', label="Threshold for Convergence")
    plt.title("Convergence Plot for Player 68")
    idx1 = np.argwhere(diffs_mean <= 1e-3).flatten()[1]
    idx2 = np.argwhere(diffs_precisions <= 1e-3).flatten()[0]

    plt.plot(idx1, min_error[idx1], 'bo', label='({}, {})'.format(float(idx1), float(min_error[idx1])))
    plt.plot(idx2, min_error[idx2], 'go', label='({}, {})'.format(float(idx2), float(min_error[idx2])))

    plt.ylabel("Absolute difference of consecutive parameters")
    plt.xlabel("Number of Message Passing Iterations")
    plt.legend()
    plt.show()


def Ex1_Autocorrs():
    G, M = load_data()
    I = 20
    n = 25
    skill_samples = gibbs_sample(G, M, I)
    plt.figure(figsize=(9, 6))
    for p in range(10):
        autocor = [pd.Series.autocorr(pd.Series(skill_samples[p, :]), lag=i) for i in range(n + 1)]
        plt.plot(autocor, linewidth=1)

    plt.axvline(10, linewidth=1, linestyle='--', color='k')
    plt.axhline(0, linewidth=0.8, color='k')
    plt.xlabel('lag');
    plt.ylabel('autocovariance coefficient')
    plt.xlim(0, n)


def Ex2B():
    G, M = load_data()
    I = 80
    DMs = MessagePassing2(G, M, I)
    threshold = 1e-3
    min_error = np.ones(I - 1) * threshold
    linspace = np.linspace(1, I, num=I - 1)
    plt.plot(linspace[5:], DMs[0][5:], 'b', label="Prior Variance 0.05")
    plt.plot(linspace[5:], DMs[1][5:], 'r', label="Prior Variance 0.5")
    plt.plot(linspace[5:], DMs[2][5:], 'y', label="Prior Variance 1")
    plt.plot(linspace[5:], DMs[3][5:], 'g', label="Prior Variance 1.5")

    plt.plot(linspace[:], min_error[:], 'orange', label="Threshold for Convergence")

    idx1 = np.argwhere(DMs[0] <= 1e-3).flatten()[0]
    idx2 = np.argwhere(DMs[1] <= 1e-3).flatten()[0]
    idx3 = np.argwhere(DMs[2] <= 1e-3).flatten()[0]
    idx4 = np.argwhere(DMs[3] <= 1e-3).flatten()[1]

    plt.plot(idx1, min_error[idx1], 'bo', label='({}, {})'.format(float(idx1), float(min_error[idx1])))
    plt.plot(idx2, min_error[idx2], 'ro', label='({}, {})'.format(float(idx2), float(min_error[idx2])))
    plt.plot(idx3, min_error[idx3], 'yo', label='({}, {})'.format(float(idx3), float(min_error[idx3])))
    plt.plot(idx4, min_error[idx4], 'go', label='({}, {})'.format(float(idx4), float(min_error[idx4])))

    plt.ylabel("Absolute difference of consecutive parameters")
    plt.xlabel("Number of Message Passing Iterations for Player 1")
    plt.legend()
    plt.show()


def ExC_Skill():
    G, M = load_data()
    I = 1000
    mean_player_skills, precision_player_skills, _, _ = MessagePassing(G, M, I)
    # Relevant player indexes
    Djokovic = 16 - 1
    Nadal = 1 - 1
    Federer = 5 - 1
    Murray = 11 - 1
    players = [Djokovic, Nadal, Federer, Murray]

    # Probability of winning against another player
    probs = np.zeros((4, 4))
    for i in range(4):  # Column
        for j in range(4):  # Row
            arg = (mean_player_skills[players[i]] - mean_player_skills[players[j]]) / (np.sqrt((
                    1 / precision_player_skills[players[i]] + 1 / precision_player_skills[players[j]])))
            prob = scipy.stats.norm.cdf(arg, 0, 1)
            probs[j, i] = prob
    print(probs)
    return probs


def ExC_Winning():
    G, M = load_data()
    I = 1000
    mean_player_skills, precision_player_skills, _, _ = MessagePassing(G, M, I)
    # Relevant player indexes
    Djokovic = 16 - 1
    Nadal = 1 - 1
    Federer = 5 - 1
    Murray = 11 - 1
    players = [Djokovic, Nadal, Federer, Murray]

    # Probability of winning against another player
    probs = np.zeros((4, 4))
    for i in range(4):  # Column
        for j in range(4):  # Row
            arg = (mean_player_skills[players[i]] - mean_player_skills[players[j]]) / np.sqrt((
                    1 + 1 / precision_player_skills[players[i]] + 1 / precision_player_skills[players[j]]))
            prob = scipy.stats.norm.cdf(arg, 0, 1)
            probs[j, i] = prob
    print(probs)
    return probs


def ExE_Empirical():
    G, M, W = load_data(1)
    # Empirical Average Predictions: Order of who won more games (indices 0-106)
    games_won = np.zeros(M)
    count = np.zeros(M)
    for p in range(M):
        count[p] = np.sum((G[:, 0] == p) * 1) + np.sum((G[:, 1] == p) * 1)
        games_won[p] = np.sum((G[:, 0] == p) * 1)
    av_games_won = np.zeros(M)
    for p in range(M):
        av_games_won[p] = games_won[p] / count[p]
    sorted_barplot(av_games_won, W)


def ExE_MP():
    G, M, W = load_data(1)
    I = 100
    # Predictions based on MP
    mean_skills_marginal, precision_skills_marginal, _, _ = MessagePassing(G, M, I)
    # Calculate winning player index probability player i wins player j in game g from performance differences
    probs = np.zeros(shape=(M, M))
    for p1 in range(M):
        for p2 in range(M):
            mean_diff = mean_skills_marginal[p1] - mean_skills_marginal[p2]
            variance = 1 / precision_skills_marginal[p1] + 1 / precision_skills_marginal[p2] + 1
            p_win_0 = scipy.stats.norm.cdf(mean_diff / np.sqrt(variance), 0, 1)
            probs[p1, p2] = p_win_0

    winning_probs = np.zeros(M)
    for p in range(M):
        winning_probs[p] = np.mean(probs[p, :])

    sorted_barplot(winning_probs, W)
    print(winning_probs)


def ExD_Gibbs():
    # Direct
    G, M, W = load_data(1)
    I = 11000
    skill_samples = gibbs_sample(G, M, I)
    Djokovic = np.argwhere(W == 'Novak-Djokovic')[0][0]
    Nadal = np.argwhere(W == 'Rafael-Nadal')[0][0]
    Federer = np.argwhere(W == 'Roger-Federer')[0][0]
    Murray = np.argwhere(W == 'Andy-Murray')[0][0]
    players = [Djokovic, Nadal, Federer, Murray]
    probs = np.zeros((4, 4))
    for i in range(4):  # Column
        for j in range(4):  # Row
            probs[j, i] = np.sum(skill_samples[players[i], 400::20] > skill_samples[players[j], 400::20]) / \
                          skill_samples[players[i], 400::20].shape[0]
    print(probs)
    return probs


def ExE_Gibbs():
    G, M, W = load_data(1)
    I = 5000
    skill_samples = gibbs_sample(G, M, I)
    probs = np.zeros((M, M))
    for i in range(len(skill_samples)):
        for j in range(len(skill_samples)):
            probs[j, i] = np.mean(scipy.stats.norm.cdf(skill_samples[i, 400::20] - skill_samples[j, 400::20]))

    av_winning_probs = np.zeros(M)
    for p in range(M):
        av_winning_probs[p] = np.sum(probs[:, p]) / (M - 1)

    I = 100
    # Predictions based on MP
    mean_skills_marginal, precision_skills_marginal, _, _ = MessagePassing(G, M, I)
    # Calculate winning player index probability player i wins player j in game g from performance differences
    probs = np.zeros(shape=(M, M))
    for p1 in range(M):
        for p2 in range(M):
            mean_diff = mean_skills_marginal[p1] - mean_skills_marginal[p2]
            variance = 1 / precision_skills_marginal[p1] + 1 / precision_skills_marginal[p2] + 1
            p_win_0 = scipy.stats.norm.cdf(mean_diff / np.sqrt(variance), 0, 1)
            probs[p1, p2] = p_win_0

    winning_probs = np.zeros(M)
    for p in range(M):
        winning_probs[p] = np.mean(probs[p, :])

    sorted_barplot(winning_probs, av_winning_probs, W)


ExE_Gibbs()
