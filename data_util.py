import random
import numpy as np

def sentence_to_np(sentence, max_sentence_len):
    # TODO add stop word to the end of all sentences (I think it is 0)
    #  haha np zeros makes sure of this
    np_sentence = np.zeros(max_sentence_len, dtype=np.int32)

    for i in range(len(sentence)):
        np_sentence[i] = sentence[i]

    return np_sentence

def get_sample(dataset, max_sentence_len):
    """ get a single sample from the dataset at random

    input:
        dataset to sample from
    output:
        (input_sentences,  - 2 sentences
        response_sentence - response sentence
        )
    """
    # choose a random movie
    movie = dataset[random.randint(0, len(dataset)-1)]
    
    # choose a random start sentence
    start_point = random.randint(0, len(movie)-2)
    sentence = sentence_to_np(movie[start_point], max_sentence_len)
    response = sentence_to_np(movie[start_point + 1], max_sentence_len)
    sentence_len = len(movie[start_point])
    response_len = len(movie[start_point+1])

    return sentence, response, sentence_len, response_len


def convert_to_id(dataset, word2id):
    """ convert dataset of words to ids

    takes dataset and converts all words to ids

    input:
        dataset - list of movies, where movies are lists of sentences
        word2id - dictionary of word -> int
    output:
        dataset with all words replaced by their integer ids
    """
    # TODO might require some pre-processing
    converted_dataset = []

    for movie in dataset:
        converted_movie = []
        for sentence in movie:
            converted_sentence = []
            for word in sentence:
                converted_sentence.append(word2id[word])

            converted_movie.append(converted_sentence)
        converted_dataset.append(converted_movie)

    return converted_dataset


def get_data_info(data, save_fname='./data/data_info.txt', 
        pre_processed=False):
    """
        Extract metadata and word dictionary from the data

        Input:
            data - List[List[List[words]]]
                 list of files where a file is: file[subtitle[word]]
        Output:
            word2id - dictionary[string word : int id]
            max_sentence_len - number of words in the longest subtitle.
    """

    max_sentence_len = 0
    word2id = {}
    word2id['<pad>'] = 0

    # TODO load from savefile
    # TODO build vocabulary?

    # extract metadata from file
    for movie in data:
        for subtitle in movie:
            # update max sentence length
            max_sentence_len = max(len(subtitle), max_sentence_len)

            for word in subtitle:
                # add word to id dictionary
                if word not in word2id and ' ' not in word:
                    word2id[word] = len(word2id)

    # TODO save to savefile

    return word2id, max_sentence_len
