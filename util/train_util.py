import random
import copy

import numpy as np
import tensorflow as tf

from util.load_util import Chat


def sentence_to_np(sentence, max_sentence_len):
    """ convert sentence to 1D np array of size max_sentence_len

    if sentence length is smaller than max_sentence_len it will be padded
    with zeros.
    """
    np_sentence = np.zeros(max_sentence_len, dtype=np.int32)

    for i in range(len(sentence)):
        np_sentence[i] = sentence[i]

    return np_sentence


def get_personas(dataset, max_sentence_len, max_conversation_len, max_persona_sentences, max_conversation_words):
    """ get two random personas from the dataset

    could be the same persona.

    Used to grab two personas for inference
    """
    persona1, _, _, _, _, _ = get_sample(
            dataset, max_sentence_len, max_conversation_len, max_conversation_words, max_persona_sentences)

    persona2, _, _, _, _, _ = get_sample(
            dataset, max_sentence_len, max_conversation_len, max_conversation_words, max_persona_sentences)

    return persona1, persona2


def get_sample(dataset, max_sentence_len,
        max_conversation_words, max_persona_sentences, word2id,
        chat_index=None, sample_index=None):
    """ get a full sample from the dataset
    input:
        dataset - dataset to sample from
        max_sentence_len - max sentence len to normalize to
        max_conversation_len - max number of sentences in a conversation to normalize to
        max_conversation_words - max number of words in a conversation to normalize to
        max_persona_sentences - max persona sentences to normalize to
        chat_index - index of chat to use. If None then pick random chat
        sample_index - index of conversation turn to use within the selected chat
                       if None then pick random turn
    output:
        persona - List[nparray[word]] - list of persona sentences
        conversation - nparray[word] - previous conversation sentences.
            sentences will start with the partner statement. Sentences will
            be separated by <pad>
        response - nparray[word] - dataset response output
    """

    if chat_index is None:
        # choose a random chat
        chat = dataset[random.randint(0, len(dataset)-1)]
    else:
        # use the selected chat
        chat = dataset[chat_index]

    # load up the persona information
    persona = chat.your_persona

    if sample_index is None:
        # choose a random piece of the conversation
        index = random.randint(0, len(chat.chat)-1)
    else:
        # use the selected piece of the conversation
        index = sample_index

    # load the previous conversation
    conversation = [word2id['<start>']]

    for i in range(0, index):
        exchange = chat.chat[i]

        # partner sentence
        for word in exchange[0]:
            conversation.append(word)
        conversation.append(word2id['<pad>'])

        # agent sentence
        for word in exchange[1]:
            conversation.append(word)
        conversation.append(word2id['<pad>'])


    # load the response
    exchange = chat.chat[index]

    for word in exchange[0]:
        conversation.append(word)
    conversation.append(word2id['<pad>'])

    conversation.append(word2id['<end>'])

    response = copy.deepcopy(exchange[1])
    response.append(word2id['<end>'])

    # convert everything to np arrays
    # TODO remove
    for i in range(len(persona)):
        sentence = persona[i]
        new_sentence = sentence_to_np(sentence, max_sentence_len)
        persona[i] = new_sentence

    # pad persona sentences
    # TODO remove
    while len(persona) < max_persona_sentences:
        pad_sentence = np.zeros(max_sentence_len, dtype=np.int32)
        persona.append(pad_sentence)
    persona = np.array(persona, dtype=np.int32)

    # TODO remove
    conversation = sentence_to_np(conversation, 
            max_conversation_words)

    response = sentence_to_np(response, max_sentence_len)

    return persona, conversation, response

def get_batch_iterator(dataset, batch_size, max_sentence_len,
        max_conversation_words,
        max_persona_sentences,
        word2id):
    """ get an iterator over consecutive batches in the eval set

    note that we may miss up to batch_size-1 samples in the dataset
    """
    personas = []
    sentences = []
    responses = []

    for sample in get_sample_iterator(
            dataset, 
            max_sentence_len,
            max_conversation_words,
            max_persona_sentences,
            word2id):

        # break out the sample
        persona, conversation, response = sample

        # add to lists
        personas.append(persona)
        sentences.append(conversation)
        responses.append(response)

        if len(personas) == batch_size:
            # pad all time dimensions to max in batch
            personas = pad_personas(personas)
            
            # convert everything to np arrays
            personas = np.array(personas)
            sentences = np.array(sentences)
            responses = np.array(responses)

            # convert to tensors
            personas = tf.constant(personas)
            sentences = tf.constant(sentences)
            responses = tf.constant(responses)

            yield personas, sentences, responses

            personas = []
            sentences = []
            responses = []


def get_sample_iterator(dataset, max_sentence_len,
        max_conversation_words, max_persona_sentences, word2id):
    class ChatMarker:
        def __init__(self, chat, chat_index):
            self.free = []
            self.chat_index = chat_index
            for i in range(len(chat.chat)):
                self.free.append(i)
    
        def get_next_index(self):
            free_index = random.randint(0, len(self.free)-1)
            sample_index = self.free.pop(free_index)
            return sample_index
        
        def is_empty(self):
            return len(self.free) == 0
    
    # set up chat markers
    free_chats = []
    for i in range(len(dataset)):
        chat = dataset[i]
        current_marker = ChatMarker(chat, i)
        free_chats.append(current_marker)
    
    # start yielding samples
    while len(free_chats) > 0:
        # choose a chat
        free_index = random.randint(0, len(free_chats)-1)
        chat_marker = free_chats[free_index]

        chat_index = chat_marker.chat_index
        sample_index = chat_marker.get_next_index()

        # remove chat if there are no samples remaining
        if chat_marker.is_empty() == True:
            del free_chats[free_index]

        yield get_sample(dataset, max_sentence_len, max_conversation_words,
                max_persona_sentences, word2id,
                chat_index, sample_index)


