import random
import numpy as np


from util.load_util import Chat


def sentence_to_np(sentence, max_sentence_len):
    # TODO add stop word to the end of all sentences (I think it is 0)
    #  haha np zeros makes sure of this
    np_sentence = np.zeros(max_sentence_len, dtype=np.int32)

    for i in range(len(sentence)):
        np_sentence[i] = sentence[i]

    return np_sentence

def get_simple_sample(dataset, max_sentence_len):
    """ get a single sample from the dataset at random

    input:
        dataset to sample from
    output:
        input sentence - 1 sentence
        response_sentence - response sentence
        sentence_len - length of the input sentence
        response_len - length of the response sentence
    """
    # choose a random chat
    chat = dataset[random.randint(0, len(dataset)-1)]
    conversation = chat.chat

    # choose a random piece of that conversation
    index = random.randint(0, len(conversation)-1)
    
    # get the exchange
    sentence = conversation[index][0]
    response = conversation[index][1]
    sentence_len = len(sentence)
    response_len = len(response)
    np_sentence = sentence_to_np(sentence, max_sentence_len)
    np_response = sentence_to_np(response, max_sentence_len)
    
    return np_sentence, np_response, sentence_len, response_len

def get_training_batch_simple(dataset, batch_size, max_sentence_len):
    """ build a batch of training data

    get batch_size worth of sentences with responses and return them as
    np arrays.

    input: 
        dataset - dataset to work with List[Conversation[]]
        batch_size - number of elements in each batch
        max_sentence_len - maximum number of words in a sentence in dataset
    output:
        sentences - List[Sentence[word]] list of sentences each of which is
            normalized to max sentence len.
        responses - List[Sentence[word]] list of sentences responding to the
            sentence in sentences with the same index each of which is 
            normalized to max sentence len.
        sentence_lens - List[int] length of each sentence in sentences
        response_lens - List[int] length of each sentence in responses
    """
    # lists
    sentences = []
    responses = []
    sentence_lens = []
    response_lens = []

    # fill lists with samples
    for i in range(batch_size):
        sentence, response, sentence_len, response_len = \
            get_simple_sample(dataset, max_sentence_len)

        sentences.append(sentence)
        responses.append(response)
        sentence_lens.append(sentence_len)
        response_lens.append(response_len)

    # convert to np arrays
    sentences = np.array(sentences)
    responses = np.array(responses)
    sentence_lens = np.array(sentence_lens)
    response_lens = np.array(response_lens)

    return sentences, responses, sentence_lens, response_lens

def get_full_sample(dataset, max_sentence_len, max_conversation_len,
        max_persona_sentences):
    """ get a full sample from the dataset at random

    input:
        dataset to sample from
        max sentence len to normalize to
        max conversation len to normalize to
        max persona sentences to normalize to
    output:
        persona - List[List[word]] - list of persona sentences
        conversation - List[word] - previous conversation sentences.
            sentences will start with the partner statement. Sentences will
            be separated by <pad>
        response - List[word] - dataset response output
        persona_sentence_lens - List[int] - length of each sentence in the persona sentences
        conversation_len - int - total length of the conversation including pads
        response_len - int - length of the response sentence
    """

    # choose a random chat
    chat = dataset[random.randint(0, len(dataset)-1)]

    # load up the persona information
    persona = chat.your_persona
    persona_lens = []
    for persona_sentence in persona:
        persona_lens.append(len(persona_sentence))

    # choose a random piece of the conversation
    index = random.randint(0, len(conversation)-1)

    # load the previous conversation
    conversation = []
    conversation_len = 0

    for i in range(0, index):
        exchange = chat.chat[i]

        # partner sentence
        for word in exchange[0]:
            conversation.append(word)
        conversation.append(0) # append id for "<pad>"
        conversation_len += len(exchange[0])

        # agent sentence
        for word in exchange[1]:
            conversation.append(word)
        conversation.append(0) # append id for "<pad>"
        conversation_len += len(exchange[1])

    # load the response
    exchange = chat.chat[index]

    for word in exchange[0]:
        conversation.append(word)
    conversation_len += len(exchange[0])

    response = exchange[1]
    response_len = len(exchange[1])

    # convert everything to np arrays

    return persona, conversation, response, persona_lens, \
        conversation_len, response_len




def get_training_batch_full(dataset, batch_size, max_sentence_len,
        max_conversation_len, max_persona_sentences):
    """ build a batch of training data. 

    get batch size worth of conversations with responses and personas
    and return them as np arrays.
    Includes all previous sentences in conversation and persona sentences.

    input:
        dataset - Dataset to work with
        batch_size - number of samples per batch
        max_sentence_len - maximum number of words per sentence
    output:
        personas - List[List[List[words]]] list of personas.
            Each persona is a list of sentences.
        sentences - List[List[words]] list of conversations. 
            Sentences separated by <pad>.
        responses - List[List[words]] list of response sentences.
        sentence_lens - List[int] length of each conversation in sentences.
        response_lens - List[int] length of each sentence in responses.
    """
    # lists
    personas = []
    sentences = []
    responses = []
    sentence_lens = []
    response_lens = []

    # fill lists with samples
    for i in range(batch_size):
        


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
    word2id = {}
    id2word = []

    def add_word(word):
        word_id = len(word2id)
        word2id[word] = word_id
        id2word.append(word)

    word2id['<pad>'] = 0
    id2word.append('<pad>')

    # TODO load from savefile
    # TODO build vocabulary?

    # extract metadata from data
    for chat in data:
        # get metadata from personas
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
        for partner_sentence, your_sentence in chat.chat:
            partner_len = len(partner_sentence)
            your_len = len(your_sentence)

            max_sentence_len = max(partner_len,  max_sentence_len)
            max_sentence_len = max(your_len,  max_sentence_len)

            for word in your_sentence + partner_sentence:
                # add word to id dictionary
                if word not in word2id and ' ' not in word:
                    add_word(word)

    # TODO save to savefile
    id2word = np.array(id2word)
    return word2id, id2word, max_sentence_len
