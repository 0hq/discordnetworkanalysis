from networkx import nx
import matplotlib.pyplot as plt
import networkx as nx
import collections
import json
from pprint import pprint
from helper_functions import *
from reaction import *
import numpy
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import re
import string
import random
import pickle


def multiple_evaluate_relationships(*arg):
    relationships = {}
    for file_name in arg:
        data = json.load(open(file_name, encoding='utf8'))
        for message in data['messages']:
            user = message['author']['name']
            if message['mentions'] != []:
                if user not in relationships:
                    relationships[user] = {}
                for mention in message['mentions']:
                    name = mention['name']
                    if name not in relationships[user]:
                        relationships[user][name] = 1
                    else:
                        relationships[user][name] += 1
    return relationships


def relationship_score(*arg):
    relationships = {}
    for file_name in arg:
        data = json.load(open(file_name, encoding='utf8'))
        for message in data['messages']:
            user = message['author']['name']
            if message['mentions'] != []:
                if user not in relationships:
                    relationships[user] = {}
                for mention in message['mentions']:
                    name = mention['name']
                    if name not in relationships[user]:
                        relationships[user][name] = 1
                    else:
                        relationships[user][name] += 1
    return relationships


def dumb_reduce(data, relationships={}):
    for bondsman in data.items():
        for lord in bondsman[1]:
            if (lord == bondsman[0]):
                continue
            if (lord, bondsman[0]) in relationships:
                relationships[(lord, bondsman[0])] += 1
            elif (bondsman[0], lord) in relationships:
                relationships[(bondsman[0], lord)] += 1
            else:
                relationships[(lord, bondsman[0])] = 1
    return relationships


def multiple_reduce(data, relationships={}):
    for bondsman in data.items():
        for lord in bondsman[1]:
            if (lord == bondsman[0]):
                continue
            if (lord, bondsman[0]) in relationships:
                relationships[(lord, bondsman[0])] += bondsman[1][lord]
            elif (bondsman[0], lord) in relationships:
                relationships[(bondsman[0], lord)] += bondsman[1][lord]
            else:
                relationships[(lord, bondsman[0])] = bondsman[1][lord]
    return relationships


def add_edges(relationships, graph):
    for mutual in relationships:
        # unadjusted is a bit squashed and excludes everyone but power users
        unadjusted_weight = relationships[mutual]
        # squaring is better, but still not great at grouping
        squared_weight = unadjusted_weight ** 2
        # the worst, don't use tanh one
        tanh_weight = numpy.tanh(unadjusted_weight)
        # current best, good at rewarding very strong connections
        exponential_capped_weight = min(100, 2 ** unadjusted_weight)
        if unadjusted_weight > 1:
            graph.add_edge(mutual[0], mutual[1],
                           weight=exponential_capped_weight)


def no_overwrite(filename, extension, i=0):
    # recursively finds a file name that does not exist to never overwrite previous saves
    path = ('{}{:d}{}'.format(filename, i, extension))
    print(path)
    if os.path.exists(path):  # if it exists, increment number next to name and continue
        i += 1
        no_overwrite(filename, extension, i)
    elif extension == '.png':  # if its a png, use the dpi argument
        plt.savefig(path, dpi=1000)
    else:
        plt.savefig(path)


def draw_graph(relationships):
    G = nx.Graph()

    add_edges(relationships, G)

    elarge = [(u, v) for (u, v, d) in G.edges(data=True)]

    # positions for all nodes
    pos = nx.spring_layout(G, k=1, seed=1)

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=90, node_color="k")

    # edges
    nx.draw_networkx_edges(
        G, pos, edgelist=elarge, width=0.1, edge_color="b")

    # labels
    nx.draw_networkx_labels(G, pos, font_size=0.3,
                            font_family="sans-serif", font_color="w")

    # no_overwrite("output\\graph", ".pdf")
    no_overwrite("output\\graph", ".png")
    # plt.axis("off")
    # plt.show()


def classify_reactions():
    reaction = condense_reaction(*all_json_files())
    noreaction = condense_nonreaction(*all_json_files())

    reaction_tokens = [word_tokenize(string) for string in reaction]
    noreaction_tokens = [word_tokenize(string) for string in noreaction]

    stop_words = stopwords.words('english')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    for tokens in reaction_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in noreaction_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    # fd = nltk.FreqDist(positive_cleaned_tokens_list)
    # fd.plot(30, cumulative=False)

    all_pos_words = get_all_words(positive_cleaned_tokens_list)

    # freq_dist_pos = FreqDist(all_pos_words)
    # print(freq_dist_pos.most_common(10))

    positive_tokens_for_model = get_tweets_for_model(
        positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(
        negative_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "Reaction")
                        for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "NoReaction")
                        for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    random.shuffle(dataset)

    train_data = dataset[:7000]
    test_data = dataset[7000:]

    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))

    print(classifier.show_most_informative_features(10))

    custom_tweet = "innovation mod present prior programming looking excited breakdown calculate"

    custom_tokens = remove_noise(word_tokenize(custom_tweet))

    print(custom_tweet, classifier.classify(
        dict([token, True] for token in custom_tokens)))

    f = open('my_classifier.pickle', 'wb')
    pickle.dump(classifier, f)
    f.close()


# unilateral_relationships = multiple_evaluate_relationships(*all_json_files())
# unilateral_relationships_small = multiple_evaluate_relationships(
#     'data\\channel_general.json')

# dumb_reduced_relationships = dumb_reduce(unilateral_relationships)
# reduced_relationships = multiple_reduce(unilateral_relationships)

# pprint(unilateral_relationships)
# pprint(dumb_reduced_relationships)
# print(json.dumps(unilateral_relationships, indent=4),
#       file=open("output\\unilateral_relationships.json", "w"))
# print(json.dumps(remap_keys(dumb_reduced_relationships), indent=4),
#       file=open("output\\udumb_reduced_relationships.json", "w"))
# print(all_json_files())

# draw_graph(dumb_reduced_relationships)
# draw_graph(reduced_relationships)
# print(condense_reaction(*all_json_files()))
# print(condense_nonreaction(*all_json_files()))
classify_reactions()
