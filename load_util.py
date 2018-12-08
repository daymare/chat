
import os
import pickle
import gzip
import shutil
import xml.etree.ElementTree as ET
from collections import Counter

from errno import ENOENT

import numpy as np


class Chat():
    def __init__(self):
        self.your_persona = []
        self.partner_persona = []
        # chat will be list of tuples (partner statement, response)
        self.chat = []


def load_dataset(dataset_filepath, pickle_file_location, load_from_pickle=True,
        save_to_pickle=True):

    if load_from_pickle == True:
        # check pickle file location exists
        if os.path.isfile(pickle_file_location) == False:
            raise Exception("pickle file does not exist!")
        
        # load from the pickle
        pickle_file = open(pickle_file_location, "rb")
        chats = pickle.load(pickle_file)
        return chats

    # open dataset file
    datafile = open(dataset_filepath, "r")
    data = datafile.read().split("\n")

    # read in personas
    chats = []
    current_chat = Chat()
    # append first persona line
    current_chat.your_persona.append(data[0].split()[3:]) 

    # skip first line since we have done it already
    # skip last line since it is empty
    for i in range(1, len(data)-1):
        line = data[i]
        words = line.split()
        if len(words) < 3:
            print("i = ", i)
            print("last 5 lines")
            print(data[i-5:i+1])
            print("line in question: ", data[i+1])

        if words[0] == "1":
            # start next chat
            chats.append(current_chat)
            current_chat = Chat()
            current_chat.your_persona.append(words[3:])

        if words[1] == "your" and words[2] == "persona:":
            current_chat.your_persona.append(words[3:])
        elif words[1] == "partner's" and words[2] == "persona:":
            current_chat.partner_persona.append(words[3:])
        else:
            # chat line add to file
            statements = line.split('\t')
            partner_statement = statements[0].split()[1:] # remove number at front
            your_statement = statements[1].split()
            current_chat.chat.append((partner_statement, your_statement))

    # append last chat to chats
    chats.append(current_chat)

    if save_to_pickle == True:
        f = open(pickle_file_location, "wb")
        pickle.dump(chats, f)

    return chats

def load_word_embeddings(fname, embedding_dim, word2id):
    if os.path.isfile(fname) == False:
        raise IOError(ENOENT, 'embedding filepath not a file!', fname)

    # initialize random word2vecs
    word2vec = np.random.normal(0, 0.05, [len(word2id), embedding_dim])
    oov = len(word2id)
    with open(fname, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            content = line.strip().split()

            # add trained word vector to our word2vec
            if content[0] in word2id:
                word2vec[word2id[content[0]]] = np.array(
                        list(map(float, content[1:])))
                oov = oov - 1

    word2vec[word2id['<pad>'], :] = 0
    print('There are %s words in vocabulary \
            and %s words out of vocabulary'
            % (len(word2id) - oov, oov))
    return word2vec


