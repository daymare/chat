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


def get_personas(dataset, word2id):
    """ get two random personas from the dataset

    could be the same persona.

    Used to grab two personas for inference
    """
    def pad_persona(persona):
        max_words = 0
        for sentence in persona:
            max_words = max(max_words, len(sentence))

        new_persona = []
        for sentence in persona:
            new_persona.append(sentence_to_np(sentence, max_words))

        return new_persona

    # get raw personas from dataset
    persona1, _, _ = get_sample(dataset, word2id)
    persona2, _, _ = get_sample(dataset, word2id)

    # pad personas
    persona1 = pad_persona(persona1)
    persona2 = pad_persona(persona2)

    return persona1, persona2


def get_sample(dataset, word2id,
        chat_index=None, sample_index=None):
    """ get a full sample from the dataset
    input:
        dataset - dataset to sample from
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

    return persona, conversation, response

def get_batch_iterator(dataset, batch_size,
        word2id):
    """ get an iterator over consecutive batches in the eval set

    note that we may miss up to batch_size-1 samples in the dataset
    """
    def pad_personas(personas):
        """ pads each persona to the max length among all persona sentences
            in the batch.
            Also pads the number of persona sentences
        """
        # persona shape: (jagged) (num_sentences, num_words)
        # personas shape: (batch_size, num_sentences, num_words)
        # after model transpose: (num_sentences, batch_size, num_words)

        # find max sentence len and max persona len
        max_words = 0
        max_persona_len = 0
        for persona in personas:
            max_persona_len = max(max_persona_len, len(persona))
            for sentence in persona:
                max_words = max(max_words, len(sentence))

        new_personas = []
        for persona in personas:
            new_persona = []
            # pad each persona sentence
            for i in range(len(persona)):
                new_persona.append(sentence_to_np(persona[i], max_words))

            # pad persona as a whole
            while len(new_persona) < max_persona_len:
                pad_sentence = np.zeros(max_words, dtype=np.int32)
                new_persona.append(pad_sentence)

            new_personas.append(new_persona)

        return new_personas

    def pad_sentences(sentences):
        """ pad batch of sentences to the max sentence length of the batch

            shape of input should be: (batch_size, sentence_lengths(jagged))
        """
        # find max sentence len
        max_len = 0
        for sentence in sentences:
            max_len = max(max_len, len(sentence))

        # pad each sentence
        new_sentences = []
        for sentence in sentences:
            new_sentences.append(sentence_to_np(sentence, max_len))

        return new_sentences


    personas = []
    sentences = []
    responses = []

    for sample in get_sample_iterator(
            dataset, 
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
            sentences = pad_sentences(sentences)
            responses = pad_sentences(responses)
            
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


def get_sample_iterator(dataset, word2id):
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

        yield get_sample(dataset, word2id,
                chat_index, sample_index)



def get_loss(predictions, responses, loss_fn):
    """ get the loss between a distribution and a given response calculated from
        the given loss function.
    """
    # TODO double check there isn't an off by one error here somewhere
    loss = 0.0
    ppl = 0.0
    for t in range(len(predictions)):
        # TODO double check responses and predictions are actually getting indexed the way we want
        sample_loss, sample_ppl = \
                loss_fn(responses[:, t], predictions[t])

        loss += sample_loss
        ppl += sample_ppl

    return loss, ppl

def calculate_hidden_cos_similarity(hidden1, hidden2, gru_over_lstm):
    """ calculate the cos similarity between two hidden states

        Notes:
            Only uses the second hidden state vector.
            Returns 0.0 if either hidden states are None
    """
    if hidden1 is None or hidden2 is None:
        return 0.0

    if gru_over_lstm is False:
        # LSTM only compare the second hidden state
        a = hidden1[1][0]
        b = hidden2[1][0]
    else:
        a = hidden1[0]
        b = hidden2[0]

    normalized_a = tf.nn.l2_normalize(a, 0)
    normalized_b = tf.nn.l2_normalize(b, 0)
    cos_similarity = tf.reduce_sum(tf.multiply(normalized_a, normalized_b))

    return cos_similarity





