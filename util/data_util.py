import random
import sys
import copy
import numpy as np
import tensorflow as tf


from util.load_util import Chat


def convert_sentence_to_id(sentence, word2id):
    converted_sentence = []

    for word in sentence.split():
        # expecting to get a lot of exceptions here from
        # running inference
        try:
            converted_sentence.append(word2id[word])
        except:
            print("exception in convert_sentence_to_id!")
            print(sentence)
            print(word)

    return np.array(converted_sentence, dtype=np.int32)

def convert_sentence_from_id(sentence, id2word):
    converted_sentence = ""
    converted_sentence += id2word[sentence[0]]

    for word_id in sentence[1:]:
        converted_sentence += " "
        word_str = id2word[word_id]
        converted_sentence += word_str

    return converted_sentence

def convert_to_id(dataset, word2id):
    """
    takes dataset and converts all words to ids

    input:
        dataset - list of chats
        word2id - dictionary of word -> int
    output:
        dataset with all words replaced by their integer ids
    """
    # TODO might require some pre-processing
    converted_dataset = []

    for chat in dataset:
        converted_chat = Chat()

        # convert self persona
        for sentence in chat.your_persona:
            converted_sentence = []
            for word in sentence:
                try:
                    converted_sentence.append(word2id[word])
                except:
                    print(sentence)
            converted_chat.your_persona.append(converted_sentence)

        # convert partner persona
        for sentence in chat.partner_persona:
            converted_sentence = []
            for word in sentence:
                converted_sentence.append(word2id[word])
            converted_chat.partner_persona.append(converted_sentence)

        # convert chat
        for partner_sentence, your_sentence in chat.chat:
            partner_converted = []
            your_converted = []

            for word in partner_sentence:
                partner_converted.append(word2id[word])

            for word in your_sentence:
                your_converted.append(word2id[word])

            converted_chat.chat.append((partner_converted, your_converted))

        converted_dataset.append(converted_chat)

    return converted_dataset

def print_chat(chat):
    """ print out the contents of a chat

    input:
        Chat object to print
    """
    print("sample chat:")
    for persona_sentence in chat.your_persona:
        print("your persona: ",persona_sentence)
    for persona_sentence in chat.partner_persona:
        print("partner_persona: ", persona_sentence)
    for partner_sentence, your_sentence in chat.chat:
        print("partner: ", partner_sentence)
        print("you: ", your_sentence)

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
    max_conversation_len = 0
    max_conversation_words = 0
    max_persona_len = 0
    word2id = {}
    id2word = []

    def add_word(word):
        word_id = len(word2id)
        word2id[word] = word_id
        id2word.append(word)


    word2id['<endpad>'] = 0
    id2word.append('<endpad>')

    word2id['<pad>'] = 1
    id2word.append('<pad>')

    word2id['<start>'] = 2
    id2word.append('<start>')

    word2id['<end>'] = 3
    id2word.append('<end>')

    # TODO load from savefile

    # extract metadata from data
    for chat in data:
        # get metadata from personas
        persona_len = len(chat.your_persona)
        max_persona_len = max(persona_len, max_persona_len)
        for sentence in chat.your_persona:
            sentence_len = len(sentence)

            max_sentence_len = max(sentence_len, max_sentence_len)

            for word in sentence:
                # add word to dictionary
                if word not in word2id and ' ' not in word:
                    add_word(word)

        for sentence in chat.partner_persona:
            sentence_len = len(sentence)

            max_sentence_len = max(sentence_len, max_sentence_len)

            for word in sentence:
                # add word to dictionary
                if word not in word2id and ' ' not in word:
                    add_word(word)

        # get metadata from chat
        conversation_len = 2 * len(chat.chat)
        max_conversation_len = max(conversation_len, max_conversation_len)
        conversation_words = 0

        for partner_sentence, your_sentence in chat.chat:
            partner_len = len(partner_sentence)
            your_len = len(your_sentence)

            conversation_words += partner_len + your_len

            max_sentence_len = max(partner_len,  max_sentence_len)
            max_sentence_len = max(your_len,  max_sentence_len)

            for word in your_sentence + partner_sentence:
                # add word to id dictionary
                if word not in word2id and ' ' not in word:
                    add_word(word)

        max_conversation_words = max(max_conversation_words,
                conversation_words)

    # account for <pads> that will be added
    max_conversation_words += max_conversation_len

    # account for <start> and <end>
    max_sentence_len += 2
    max_conversation_words += 2

    # TODO save to savefile
    id2word = np.array(id2word)
    return word2id, id2word, max_sentence_len, max_conversation_len, \
            max_conversation_words, max_persona_len
