import os
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
import re
import string
import random
import json
from datetime import datetime
from matplotlib import pyplot as plt
from collections import Counter


def remap_keys(mapping):
    return [{'key': k, 'value': v} for k, v in mapping.items()]


def all_json_files():
    directory = os.fsencode(str("data"))
    return [os.path.join(directory, file) for file in os.listdir(
        directory) if os.fsdecode(file).endswith(".json")]


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

# print(all_json_files())


def response_times_relationships(*arg):
    reply_timestamps = {}
    reference_list = []
    for file_name in arg:
        data = json.load(open(file_name, encoding='utf8'))
        message_list = data['messages']
        message_list.reverse()
        for message in message_list:
            if message['id'] in reference_list:
                reply_timestamps[message['id']] = [message['timestamp']]
            if message['type'] == "Reply":
                reply_timestamps[message['id']] = [
                    message['timestamp'], message['reference']['messageId']]
                reference_list.append(message['reference']['messageId'])
    return reply_timestamps


def subtract_times(t1, t2):
    t1 = datetime.strptime(t1,
                           str('%Y-%m-%dT%H:%M:%S.%f+00:00'))
    t2 = datetime.strptime(t2,
                           str('%Y-%m-%dT%H:%M:%S.%f+00:00'))
    result = t1 - t2
    return result.seconds


def calculate_response_times(*arg):
    k = []
    times = response_times_relationships(*arg)
    for response in times:
        if len(times[response]) == 2:
            try:
                k.append(subtract_times(times[response][0],
                                        times[times[response][1]][0]))
            except:
                continue
        else:
            continue
    return sorted(k)


response_times = calculate_response_times(*all_json_files())

print(response_times)


def statistics_response_times(times):
    n_num = times
    n = len(n_num)
    n_num.sort()
    sum_num = sum(times)

    if n % 2 == 0:
        median1 = n_num[n//2]
        median2 = n_num[n//2 - 1]
        median = (median1 + median2)/2
    else:
        median = n_num[n//2]

    print("Average is: " + str(sum_num/n))

    print("Median is: " + str(median))

# print(subtract_times("2020-11-25T06:06:44.68+00:00", "2020-12-01T23:57:48.98+00:00"))


plt.hist(Counter(response_times), bins=1000)
plt.gca().set(title='Frequency of Reply Timeout',
              ylabel='Frequency', xlabel='Seconds Elapsed')
plt.show()
statistics_response_times(response_times)
