from networkx import nx
import matplotlib.pyplot as plt
import networkx as nx
import collections
import json
from pprint import pprint
from helper_functions import *
import numpy


def dangerous_evaluate_relationships(data, relationships={}):
    for message in data['messages']:
        user = message['author']['name']
        if message['mentions'] != []:
            if user not in relationships:
                relationships[user] = {}
                for mention in message['mentions']:
                    name = mention['name']
                    print(name)
                    pprint(mention)
                    if name not in relationships[user]:
                        relationships[user][name] = 1
                    else:
                        relationships[user][name] = 999
    return relationships


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


def dumb_reduce_efficient(data, relationships={}):
    for bondsman in data.items():
        # print(bondsman)
        for lord in bondsman[1]:
            # print(lord)
            if frozenset({lord, bondsman[0]}) in relationships:
                relationships[frozenset({lord, bondsman[0]})] += 1
            else:
                relationships[frozenset({lord, bondsman[0]})] = 1
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


def add_edges(relationships, graph):
    for mutual in relationships:
        unadjusted_weight = relationships[mutual]
        squared_weight = unadjusted_weight ** 2
        tanh_weight = numpy.tanh(unadjusted_weight)
        exponential_capped_weight = min(100, 2 ** unadjusted_weight)
        graph.add_edge(mutual[0], mutual[1],
                       weight=exponential_capped_weight)


def no_overwrite(filename, extension, i=0):
    path = ('{}{:d}{}'.format(filename, i, extension))
    print(path)
    if os.path.exists(path):
        i += 1
        no_overwrite(filename, extension, i)
    elif extension == '.png':
        plt.savefig(path, dpi=1000)
    else:
        plt.savefig(path)


def draw_graph(relationships):
    G = nx.Graph()

    add_edges(relationships, G)

    elarge = [(u, v) for (u, v, d) in G.edges(data=True)]
    # esmall = [(u, v) for (u, v, d) in G.edges(data=True) if d["weight"] <= 0.5]

    pos = nx.spring_layout(G, 1)  # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=90, node_color="k")

    # edges
    # nx.draw_networkx_edges(G, pos, edgelist=elarge, width=1)
    nx.draw_networkx_edges(
        G, pos, edgelist=elarge, width=0.1, edge_color="b")

    # labels
    nx.draw_networkx_labels(G, pos, font_size=0.3,
                            font_family="sans-serif", font_color="w")

    plt.axis("off")
    # plt.savefig("output\\graph.png", dpi=1000)
    # plt.savefig("output\\graph.pdf")
    no_overwrite("output\\graph", ".pdf")
    no_overwrite("output\\graph", ".png")

    # plt.show()


unilateral_relationships = multiple_evaluate_relationships(*all_json_files())
unilateral_relationships_small = multiple_evaluate_relationships(
    'data\\channel_general.json')

dumb_reduced_relationships = dumb_reduce(unilateral_relationships)

# pprint(unilateral_relationships)
# pprint(dumb_reduced_relationships)
# print(json.dumps(unilateral_relationships, indent=4),
#       file=open("output\\unilateral_relationships.json", "w"))
# print(json.dumps(remap_keys(dumb_reduced_relationships), indent=4),
#       file=open("output\\udumb_reduced_relationships.json", "w"))
# print(all_json_files())

draw_graph(dumb_reduced_relationships)
