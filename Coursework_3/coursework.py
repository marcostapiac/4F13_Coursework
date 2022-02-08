import numpy as np
import scipy.io as sio
from matplotlib import pyplot as plt
from tqdm import tqdm
from labellines import labelLines
from bmm import BMM
from lda import LDA


def load_data():
    # set seed for reproducibility
    np.random.seed(0)
    # load data
    data = sio.loadmat('kos_doc_data.mat')
    # Array containing the names of each player
    A = data['A']
    B = data['B']
    V = []
    for arr in data['V']:
        V.append(arr[0][0])
    V = np.array(V)
    return A, B, V


def sorted_barplotA(vals, labels, figname, title):
    # Plot a barplot
    sorted_indices = np.argsort(vals)[::-1][:20]
    height = 0.01
    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(6.27, 11.69, forward=True)
    xx = np.linspace(20, 0, 20)  # Plot 20 largest probabilities
    horizontal_vals = vals[sorted_indices]
    plt.barh(xx, horizontal_vals, label="Largest 20 ML Probabilities")
    plt.yticks(np.linspace(0, 20, 20) + height / 2, labels=labels[sorted_indices][::-1], fontsize=12)
    print(labels[sorted_indices])
    plt.xlabel("Probability", fontsize=16)
    plt.ylabel("Word", fontsize=16, labelpad=4)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.savefig(figname)
    plt.show()


def exA(figname="Figure1_Word_Barplot.png"):
    # Multinomial ML over training words in all documents
    A, _, V = load_data()
    distinct_num_words = V.shape[0]
    num_words = np.sum(A[:, 2])
    pi_ML = [0] * (distinct_num_words)
    for word_id in np.unique(A[:, 1]):
        args = np.where(A[:, 1] == word_id)[0]
        pi_ML[word_id - 1] = np.sum(A[args, 2])
    pi_ML = np.array(pi_ML)
    pi_ML = pi_ML / num_words
    sorted_barplotA(pi_ML, labels=V, figname=figname, title="Maximum Likelihood estimates for words")
    print(max(pi_ML))
    return pi_ML


# exA()
def sorted_barplot_compare(vals, labels, figname, title, compare_vals):
    # Plot a barplot
    sorted_indices = np.argsort(vals)[::-1][:20]
    height = 0.2
    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(6.27, 11.69, forward=True)
    xx = np.linspace(20, 0, 20)  # Plot 20 largest probabilities
    horizontal_vals = vals[sorted_indices]
    compare_horizontal_vals = compare_vals[sorted_indices]
    plt.barh(xx + height, horizontal_vals, height=height, label="Largest 20 Predictive Probabilities")
    plt.barh(xx, compare_horizontal_vals, height=height, label="Largest 20 ML Probabilities")
    plt.yticks(np.linspace(0, 20, 20) + height / 2, labels=labels[sorted_indices][::-1], fontsize=12)
    print(labels[sorted_indices])
    plt.xlabel("Probability", fontsize=16)
    plt.ylabel("Word", fontsize=16, labelpad=4)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.savefig(figname)
    plt.show()


def sorted_barplot_B(vals, labels, figname, title):
    # Plot a barplot
    sorted_indices = np.argsort(vals)[::-1][:20]
    height = 0.01
    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(6.27, 11.69, forward=True)
    xx = np.linspace(20, 0, 20)  # Plot 20 largest probabilities
    horizontal_vals = vals[sorted_indices]
    plt.barh(xx, horizontal_vals, height=height, label="Largest 20 Predictive Probabilities")
    plt.yticks(np.linspace(0, 20, 20) + height / 2, labels=labels[sorted_indices][::-1], fontsize=12)
    print(labels[sorted_indices])
    plt.xlabel("Probability", fontsize=16)
    plt.ylabel("Word", fontsize=16, labelpad=4)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.savefig(figname)
    plt.show()


def sorted_barplot_uniform(vals, labels, figname, title, uniform_prob):
    # Plot a barplot
    sorted_indices = np.argsort(vals)[::-1][:20]
    height = 0.5
    fig = plt.figure(tight_layout=True)
    fig.set_size_inches(6.27, 11.69, forward=True)
    xx = np.linspace(20, 0, 20)  # Plot 20 largest probabilities
    horizontal_vals = vals[sorted_indices]
    plt.barh(xx, horizontal_vals, height=height, label="Largest 20 Predictive Probabilities")
    plt.yticks(np.linspace(0, 20, 20) + height / 2, labels=labels[sorted_indices][::-1], fontsize=12)
    plt.plot(np.ones(20) * uniform_prob, np.linspace(-1, 22, 20), 'orange',
             label="Theoretical Uniform Probability $\\approx " + str(round(uniform_prob, 5)) + "$")
    plt.ylim(-1, 21)
    print(labels[sorted_indices])
    plt.xlabel("Probability", fontsize=16)
    plt.ylabel("Word", fontsize=16, labelpad=4)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.savefig(figname)
    plt.show()


def exB(alpha, figname):
    A, B, V = load_data()
    unique_A_words = np.unique(A[:, 1])
    unique_B_words = np.unique(B[:, 1])
    unique_B_words_not_in_A = np.array([word_id for word_id in unique_B_words if word_id not in unique_A_words])
    num_words_in_A = np.sum(A[:, 2])
    distinct_num_words = V.shape[0]
    distinct_num_words_A = unique_A_words.shape[0]
    distinct_num_words_B_not_in_A = unique_B_words_not_in_A.shape[0]
    assert (distinct_num_words_A + distinct_num_words_B_not_in_A == distinct_num_words)
    # Predictive probabilities under symmetric Dirichlet prior
    pi = [0] * distinct_num_words
    for word_id in unique_A_words:
        args = np.where(A[:, 1] == word_id)[0]
        word_count = np.sum(A[args, 2])
        pi[word_id - 1] = (alpha + word_count) / (alpha * distinct_num_words + num_words_in_A)
    for word_id in unique_B_words_not_in_A:
        pi[word_id - 1] = alpha / (distinct_num_words * alpha + num_words_in_A)
    pi = np.array(pi)
    if alpha == 0:
        pi_ML = exA(figname="")
        sorted_barplot_compare(pi, labels=V, figname=figname,
                               title="Predictive Probabilities for $ \\alpha = " + str(int(alpha)) + "$",
                               compare_vals=pi_ML)
    elif alpha > 1e6:
        uniform_prob = 1 / distinct_num_words
        sorted_barplot_uniform(pi, labels=V, figname=figname,
                               title="Predictive Probabilities for $ \\alpha = 1e10 $", uniform_prob=uniform_prob)
    else:
        sorted_barplot_B(pi, labels=V, figname=figname,
                         title="Predictive Probabilities for $ \\alpha = " + str(int(alpha)) + "$")


# exB(1e10, "Figure2_LargeAlpha_Word_Barplot.png")


def plot_alphas(xvals, alpha, yvals, figname):
    max_x = max(xvals)
    num_alphas = xvals.shape[0]
    plt.plot(xvals, yvals, label="Per word perplecxity$=" + str(yvals[0]) + "$")
    plt.xlabel("Document Number", fontsize=14)
    plt.ylabel("Per Word Perplexity", fontsize=14)
    plt.legend()
    plt.title("Per Word Perplexity of Test Documents for $\\alpha= " + str(alpha) + "$", fontsize=14)
    plt.savefig(figname)
    plt.show()


def plot_lls(xvals, yvals, figname):
    plt.rcParams["figure.figsize"] = (7, 5)
    min_y = min(yvals)
    num_alphas = xvals.shape[0]
    plt.plot(xvals, yvals, label="Log Likelikelihood")
    plt.plot(xvals, np.ones(num_alphas) * min_y, label="Minimum LL $= " + str(round(min_y, 3)) + "$")
    plt.xlabel("Concentration Value", fontsize=14)
    plt.ylabel("Log Likelihood", fontsize=14)
    plt.legend()
    plt.title("Log Probability of Test Documents", fontsize=14)
    plt.savefig(figname)
    plt.show()


def plot_pps(xvals, yvals, figname):
    plt.rcParams["figure.figsize"] = (7, 5)
    max_y = max(yvals)
    num_alphas = xvals.shape[0]
    plt.plot(xvals, yvals, label="Per Word Perplexity")
    plt.plot(xvals, np.ones(num_alphas) * max_y, label="Maximum Perplexity $= " + str(round(max_y, 3)) + "$")
    plt.xlabel("Concentration Value", fontsize=14)
    plt.ylabel("Perplexity", fontsize=14)
    plt.legend()
    plt.title("Per word perplexity of Test Documents", fontsize=14)
    plt.savefig(figname)
    plt.show()


def exC1(alpha_max, num_alphas):
    A, B, V = load_data()
    # Obtain predicitive probabilities
    doc_2001_wordIDs = np.unique(np.array(B[np.where(B[:, 0] == 2001)[0], 1]))
    doc_2001_counts = np.array(B[np.where(B[:, 0] == 2001)[0], 2])
    unique_A_words = np.unique(A[:, 1])
    unique_B_words = np.unique(B[:, 1])
    unique_B_words_not_in_A = np.array([word_id for word_id in unique_B_words if word_id not in unique_A_words])
    num_words_2001 = np.sum(doc_2001_counts)
    num_words_in_A = np.sum(A[:, 2])
    distinct_num_words = V.shape[0]
    distinct_num_words_A = unique_A_words.shape[0]
    distinct_num_words_B_not_in_A = unique_B_words_not_in_A.shape[0]
    assert (distinct_num_words_A + distinct_num_words_B_not_in_A == distinct_num_words)
    alphas = np.array([5])  # np.linspace(0, alpha_max, num=num_alphas)
    lls = []
    pps = []  # Per word perplexities
    for alpha in alphas:
        # Predictive probabilities under symmetric Dirichlet prior
        pi = np.zeros(shape=(distinct_num_words, 2))
        i = 0
        for word_id in unique_A_words:
            args = np.where(A[:, 1] == word_id)[0]
            word_count = np.sum(A[args, 2])
            pi[i, 0] = word_id
            pi[i, 1] = (alpha + word_count) / (alpha * distinct_num_words + num_words_in_A)
            i += 1
        for word_id in unique_B_words_not_in_A:
            pi[i, 0] = word_id
            pi[i, 1] = alpha / (distinct_num_words * alpha + num_words_in_A)
            i += 1
        # Get probabilities for words in doc 2001
        doc_2001_predProbs = np.zeros(doc_2001_wordIDs.shape[0])
        for j in range(doc_2001_wordIDs.shape[0]):
            arg = np.where(pi[:, 0] == doc_2001_wordIDs[j])[0][0]
            doc_2001_predProbs[j] = pi[arg, 1]
        # Log likelihood of document
        ll = np.sum(doc_2001_counts.transpose().dot(np.log(doc_2001_predProbs)))
        lls.append(ll)
        pps.append(np.exp(-ll / num_words_2001))
    print(pps)
    lls = np.array(lls)
    pps = np.array(pps)


# plot_lls(xvals=alphas, yvals=lls, figname="Figure3_Test2001_PredictiveProbabilities2.png")
# plot_pps(xvals=alphas, yvals=pps, figname="Figure3_Test2001_Perplexities2.png")

# exC1(alpha_max=0.1, num_alphas=100)

# exC1(alpha_max=40000, num_alphas=100)


def plot_pps_exC2(xvals, alpha, yvals, figname):
    plt.rcParams["figure.figsize"] = (10, 5)
    max_y = max(yvals)
    num_alphas = xvals.shape[0]
    if (alpha > 10):
        plt.ylim(6906 / 2, 6906 * 1.5)
    plt.plot(xvals, yvals, label="Per Word Perplexity")
    plt.plot(xvals, np.ones(num_alphas) * max_y, label="Maximum PP $= " + str(round(max_y, 3)) + "$")
    plt.xlabel("Document Number", fontsize=14)
    plt.ylabel("Perplexity", fontsize=14)
    plt.legend()
    plt.title("Per word perplexity of Test Documents for $\\alpha = " + str(alpha) + "$", fontsize=14)
    plt.savefig(figname)
    plt.show()


def exC2(alpha, figname):
    A, B, V = load_data()
    unique_A_words = np.unique(A[:, 1])
    unique_B_words = np.unique(B[:, 1])
    unique_B_words_not_in_A = np.array([word_id for word_id in unique_B_words if word_id not in unique_A_words])
    num_words_in_A = np.sum(A[:, 2])
    distinct_num_words = V.shape[0]
    distinct_num_words_A = unique_A_words.shape[0]
    distinct_num_words_B_not_in_A = unique_B_words_not_in_A.shape[0]
    assert (distinct_num_words_A + distinct_num_words_B_not_in_A == distinct_num_words)
    pi = np.zeros(shape=(distinct_num_words, 2))
    i = 0
    for word_id in unique_A_words:
        args = np.where(A[:, 1] == word_id)[0]
        word_count = np.sum(A[args, 2])
        pi[i, 0] = word_id
        pi[i, 1] = (alpha + word_count) / (alpha * distinct_num_words + num_words_in_A)
        i += 1
    for word_id in unique_B_words_not_in_A:
        pi[i, 0] = word_id
        pi[i, 1] = alpha / (distinct_num_words * alpha + num_words_in_A)
        i += 1
    pps = []  # Per word perplexities
    # For every doc in B
    for doc_id in tqdm(np.unique(B[:, 0])):
        doc_wordIDs = np.array(B[np.where(B[:, 0] == doc_id)[0], 1])
        doc_counts = np.array(B[np.where(B[:, 0] == doc_id)[0], 2])
        num_words_doc = np.sum(doc_counts)
        # Get probabilities for words in doc 2001
        doc_predProbs = np.zeros(doc_wordIDs.shape[0])
        for i in range(doc_wordIDs.shape[0]):
            arg = np.where(pi[:, 0] == doc_wordIDs[i])[0][0]
            doc_predProbs[i] = pi[arg, 1]
        # Log likelihood of document
        ll = np.sum(doc_counts.transpose().dot(np.log(doc_predProbs)))
        pp = np.exp(-ll / num_words_doc)
        pps.append(pp)
    pps = np.array(pps)
    print(pps)
    xvals = np.linspace(1, np.unique(B[:, 0]).shape[0], np.unique(B[:, 0]).shape[0])
    plot_pps_exC2(xvals=xvals, alpha=alpha, yvals=pps, figname=figname)


# exC2(alpha=1e100, figname="Figure_3_DocumentPerplexities_LargeAlpha.png")
# exC2(alpha=5, figname="Figure_3_DocumentPerplexities_MediumAlpha.png")
# exC2(alpha=0.001, figname="Figure_3_DocumentPerplexities_SmallAlpha.png")
# exC2(alpha=0.0, figname="Figure_3_DocumentPerplexities_ML.png")


def exC3(alpha_max, num_alphas):
    A, B, V = load_data()
    # Obtain predictive probabilities
    doc_B_wordIDs = np.unique(B[:, 1])  # All the unique words over all the docs in B
    doc_B_counts = [0] * doc_B_wordIDs.shape[0]  # Total count for each word in B
    for i in range(doc_B_wordIDs.shape[0]):
        doc_B_counts[i] = np.sum(np.array(B[np.where(B[:, 1] == doc_B_wordIDs[i])[0], 2]))
    doc_B_counts = np.array(doc_B_counts)
    print(np.count_nonzero(doc_B_counts))
    assert (np.sum(doc_B_counts) == np.sum(B[:, 2]))
    unique_A_words = np.unique(A[:, 1])
    unique_B_words_not_in_A = np.array([word_id for word_id in doc_B_wordIDs if word_id not in unique_A_words])
    num_words_doc = np.sum(doc_B_counts)
    num_words_in_A = np.sum(A[:, 2])
    distinct_num_words = V.shape[0]
    distinct_num_words_A = unique_A_words.shape[0]
    distinct_num_words_B_not_in_A = unique_B_words_not_in_A.shape[0]
    assert (distinct_num_words_A + distinct_num_words_B_not_in_A == distinct_num_words)
    alphas = np.array([5])  # np.linspace(1e-16, alpha_max, num=num_alphas)
    lls = []
    pps = []  # Per word perplexities
    for alpha in tqdm(alphas):
        # Predictive probabilities under symmetric Dirichlet prior
        pi = np.zeros(shape=(distinct_num_words, 2))
        i = 0
        for word_id in unique_A_words:
            args = np.where(A[:, 1] == word_id)[0]
            word_count = np.sum(A[args, 2])
            pi[i, 0] = word_id
            pi[i, 1] = (alpha + word_count) / (alpha * distinct_num_words + num_words_in_A)
            i += 1
        for word_id in unique_B_words_not_in_A:
            pi[i, 0] = word_id
            pi[i, 1] = alpha / (distinct_num_words * alpha + num_words_in_A)
            i += 1
        # Get probabilities for words in doc 2001
        doc_2001_predProbs = np.zeros(doc_B_wordIDs.shape[0])
        for j in range(doc_B_wordIDs.shape[0]):
            arg = np.where(pi[:, 0] == doc_B_wordIDs[j])[0][0]
            doc_2001_predProbs[j] = pi[arg, 1]
        # Log likelihood of document
        ll = np.sum(doc_B_counts.transpose().dot(np.log(doc_2001_predProbs)))
        lls.append(ll)
        pps.append(np.exp(-ll / num_words_doc))
    pps = np.array(pps)
    print(pps)
    plot_pps(xvals=alphas, yvals=pps, figname="Figure_3_DocumentB_Perplexity.png")


# exC3(alpha_max=40000, num_alphas=100)


def plot_mix_prop(mix_prop_evols, num_iters_gibbs, K, alpha, gamma, figname):
    fig, ax = plt.subplots()
    cutoff = 0.003
    for i in range(K):
        label = str(i + 1) if mix_prop_evols[-1, i] > cutoff else ""
        ax.plot(np.linspace(0, num_iters_gibbs, num_iters_gibbs), mix_prop_evols[:, i], label=label)

    plt.grid()
    plt.xlabel("Gibbs Iteration Number", fontsize=14)
    plt.ylabel("Mixing Proportions", fontsize=14)
    plt.title("Mixing Proportions for " + str(K) + " categories for $ \\alpha = " + str(alpha) + "$, $\gamma = " + str(
        gamma) + "$", fontsize=14)
    labelLines(ax.get_lines())
    plt.savefig(figname)
    plt.show()


def plot_convergence(threshold, diff_mix_prop_evols, num_iters_gibbs, K, alpha, gamma, figname):
    fig, ax = plt.subplots()
    xvals = np.linspace(0, num_iters_gibbs - 1, num_iters_gibbs - 1)

    for i in range(K):
        cutoff = 0.8e-3
        label = "Category " + str(i + 1) if diff_mix_prop_evols[-1, i] > cutoff else ""
        ax.plot(xvals, diff_mix_prop_evols[:, i], label=label)
    ax.plot(xvals, np.ones(num_iters_gibbs - 1) * threshold,
            label="Convergence Threshold $= " + str(threshold) + "$")
    plt.xlabel("Gibbs Iteration Number", fontsize=14)
    plt.ylabel("Absolute difference in consecutive values", fontsize=14)
    plt.title("Gibbs Convergence Plot for " + str(K) + " categories", fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig(figname)
    plt.show()


def sorted_barplot(vals, labels, figname, title, K):
    # Plot a barplot
    sorted_indices = np.argsort(vals)[::-1][:K]
    height = 0.01
    xx = np.linspace(K, 0, K)  # Plot 20 largest probabilities
    fig, ax = plt.subplots()
    fig.set_size_inches(18, 11.69, forward=True)
    horizontal_vals = vals[sorted_indices]
    ax.barh(xx, horizontal_vals, label="Mixing Proportions")
    plt.yticks(np.linspace(0, K, K) + height / 2, labels=labels[sorted_indices][::-1]+1, fontsize=12)
    plt.xlabel("Probability", fontsize=16)
    plt.ylabel("Category", fontsize=16, labelpad=4)
    plt.title(title, fontsize=16)
    for i in range(len(vals)):
        plt.text(horizontal_vals[i] + height / 2, xx[i], round(horizontal_vals[i], 5))
    plt.legend()
    plt.savefig(figname)
    plt.show()


def exD(alpha=5, gamma=0.1, seed=27, K=20):
    A, B, V = load_data()
    num_iters_gibbs = 50
    perplexity, mult_weights, mix_prop_evols, diff_mix_prop_evols = BMM(A, B, K, alpha, gamma,
                                                                        num_iters_gibbs=num_iters_gibbs, seed=seed)
    V = np.array([i for i in range(K)])
    print(mix_prop_evols[-1, :])
    print(perplexity)
    sorted_barplot(mix_prop_evols[-1, :], labels=V, figname="Figure_5_SortedBarplot_K20_Seed1.png",title="Mixing Proportions for $ \\alpha = " + str(alpha) + "$, $\gamma = " + str(gamma) + "$", K=K)
    plot_mix_prop(mix_prop_evols, num_iters_gibbs, K, alpha, gamma, figname="Figure_5_Convergence_Plots_K20_Seed1.png")
    plot_convergence(2.5e-3, diff_mix_prop_evols, num_iters_gibbs, K, alpha, gamma, figname="Figure_5_AbsDiff_Convergence_Plots_K20_Seed1.png")


#exD(K=20, seed=100) #2102.488120212345
#exD(K=3, seed=1) #2235.654817348998
#exD(K=20, seed=10) #2113.5772440441797
#exD(K=20, seed=1) # 2084.9225471841773

def plot_word_entropies(word_entropies, num_iters, K, alpha, gamma, figname):
    fig, ax = plt.subplots()
    xvals = np.linspace(0, num_iters, num_iters)
    cutoff = 0.003
    for i in range(K):
        label = str(i + 1)
        ax.plot(xvals, word_entropies[:, i], label=label)
    labelLines(ax.get_lines())
    plt.xlabel("Gibbs Iteration Number", fontsize=14)
    plt.ylabel("Word Entropy", fontsize=14)
    plt.title("Word Entropies for " + str(K) + " categories", fontsize=14)
    plt.grid()
    plt.savefig(figname)
    plt.show()


def exE(figname, alpha=5, gamma=0.1):
    A, B, V = load_data()
    K = 20
    num_iters_gibbs = 50
    perplexity, mult_weights, mix_prop_evols, diff_mix_prop_evols, word_entropies, av_mix_prop_evols = LDA(A, B, K,
                                                                                                           alpha, gamma,
                                                                                                           num_iters_gibbs=num_iters_gibbs)
    print(perplexity)
    plot_mix_prop(av_mix_prop_evols, num_iters_gibbs, K, alpha, gamma,
                  figname="Figure_6_Av_Convergence_Plots_K20_Doc1.png")
    plot_mix_prop(mix_prop_evols[0], num_iters_gibbs, K, alpha, gamma, figname="Figure_6_Convergence_Plots_K20_Doc1.png")
    plot_word_entropies(word_entropies, num_iters_gibbs, K, alpha, gamma, figname="Figure_6_WordEntropies_K20.png")
    plot_mix_prop(mix_prop_evols[0], num_iters_gibbs, K, alpha, gamma, figname="Figure_6_Convergence_Plots_K20_Doc1.png")
    plot_mix_prop(mix_prop_evols[16], num_iters_gibbs, K, alpha, gamma, figname="Figure_6_Convergence_Plots_K20_Doc17.png")
    plot_mix_prop(mix_prop_evols[899], num_iters_gibbs, K, alpha, gamma, figname="Figure_6_Convergence_Plots_K20_Doc900.png")

    plot_convergence(5e-3, diff_mix_prop_evols[0], num_iters_gibbs, K, alpha, gamma, figname="Figure_6_AbsDiff_Convergence_Plots_K20_Doc1.png")


exE(figname="")
