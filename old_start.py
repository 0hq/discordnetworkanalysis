import networkx as nx
import plotly.graph_objects as go
import nltk
from nltk.book import *


def lexical_diversity(text):
    return len(set(text)) / len(text)


def percentage(count, total):
    return 100 * count / total


def example_frequency_distribution(text):
    fdist = FreqDist(text)
    print(list(fdist))
    # fdist.most_common(50)
    # for sample in fdist:
    #     print(sample)
    # sample[1] = sample[1] / fdist.N()
    fdist.plot(50, cumulative=True)
    fdist.plot(50)


# # example_frequency_distribution(text1)
# emma = nltk.corpus.gutenburg.words('austen-emma.txt')
# len(emma)


G = nx.random_geometric_graph(200, 0.125)
