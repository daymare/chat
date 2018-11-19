
import os
import pickle
import gzip
import shutil
import xml.etree.ElementTree as ET
from collections import Counter

from errno import ENOENT

import numpy as np


def extract_file_contents(filename):
    """ extract list of subtitles from the given file

    input:
        fileanme - file path to xml document to parse
            must be in the opensubtitles format
    output:
        List[List[words]] - list of subtitles.
            Each subtitle is a list of words
            including punctuation as individual words
    """
    if not os.path.isfile(filename):
        raise IOError(ENOENT, 'not a file!', filename)

    xml_tree = ET.parse(filename)
    root = xml_tree.getroot()

    subtitles = []
    # for each subtitle in the document
    for s in root:
        subtitle = []
        # get all the words
        for word in s.iter('w'):
            subtitle.append(word.text)
        subtitles.append(subtitle)

    return subtitles

def gunzip(filepath):
    with gzip.open(filepath, "rb") as f_in:
        with open(filepath[:-3], "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

def extract_folder_contents(foldername):
    """ recursivwely extract subtitles from all files in a folder

    input:
        foldername - filepath to folder that we want to get files from

    output:
        List[List[List[words]]] - list of file contents 
            which is each a list of subtitles
            which is each a list of words (strings)
        effectively List[files(subtitles(words))]
    """
    file_contents = []
    # recurse down directories
    files = os.listdir(foldername)
    for f in os.listdir(foldername):
        fpath = os.path.join(foldername, f)
        if os.path.isdir(fpath):
            # recurse down directory
            file_contents = file_contents + extract_folder_contents(fpath)
        elif f.endswith(".gz") and fpath[:-3] not in files:
            # unzip .gz
            gunzip(fpath)

            # process that file
            file_contents.append(extract_file_contents(fpath[:-3]))
        elif f.endswith(".xml"):
            file_contents.append(extract_file_contents(fpath))

    return file_contents

def load_dataset(dataset_folder, pickle_file_location, load_from_pickle=True, 
        save_to_pickle=True):
    """
        load the dataset either from the folder or from a pickle file
    """
    if load_from_pickle == True:
        # check pickle file location exists
        if os.path.isfile(pickle_file_location) == False:
            raise Exception("pickle file does not exist!")

        # load from the pickle
        pickle_file = open(pickle_file_location, "rb")
        dataset = pickle.load(pickle_file)
        return dataset
    else:
        # load from dataset folder
        dataset = extract_folder_contents(dataset_folder)

        # save to pickle file
        if save_to_pickle == True:
            f = open(pickle_file_location, "wb")
            pickle.dump(dataset, f)

        return dataset


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


if __name__ == "__main__":
    folder_filepath = "./data/OpenSubtitles/en/2002/"
    filename = "3443_211529_278309_an_american_in_canada.xml"
    file_filepath = os.path.join(folder_filepath, filename);
    
    # test all the functions
    file_contents = extract_file_contents(file_filepath)
    #print(file_contents)

    folder_contents = extract_folder_contents(folder_filepath)
    print(folder_contents)

