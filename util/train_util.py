import random

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
        persona_sentence_lens - List[int] - length of each sentence in the persona sentences
        conversation_len - int - total length of the conversation including pads
        response_len - int - length of the response sentence
    """

    if chat_index is None:
        # choose a random chat
        chat = dataset[random.randint(0, len(dataset)-1)]
    else:
        # use the selected chat
        chat = dataset[chat_index]

    # load up the persona information
    persona = chat.your_persona
    persona_lens = []
    for persona_sentence in persona:
        persona_lens.append(len(persona_sentence))

    if sample_index is None:
        # choose a random piece of the conversation
        index = random.randint(0, len(chat.chat)-1)
    else:
        # use the selected piece of the conversation
        index = sample_index

    # load the previous conversation
    conversation = [word2id['<start>']]
    conversation_len = 1

    for i in range(0, index):
        exchange = chat.chat[i]

        # partner sentence
        for word in exchange[0]:
            conversation.append(word)
        conversation.append(word2id['<pad>'])
        conversation_len += len(exchange[0]) + 1

        # agent sentence
        for word in exchange[1]:
            conversation.append(word)
        conversation.append(word2id['<pad>'])
        conversation_len += len(exchange[1]) + 1


    # load the response
    exchange = chat.chat[index]

    for word in exchange[0]:
        conversation.append(word)
    conversation.append(word2id['<pad>'])
    conversation_len += len(exchange[0]) + 1

    conversation.append(word2id['<end>'])
    conversation_len += 1

    response = copy.deepcopy(exchange[1])
    response.append(word2id['<end>'])
    response_len = len(exchange[1]) + 1

    # convert everything to np arrays
    for i in range(len(persona)):
        sentence = persona[i]
        new_sentence = sentence_to_np(sentence, max_sentence_len)
        persona[i] = new_sentence

    # pad persona sentences
    while len(persona) < max_persona_sentences:
        pad_sentence = np.zeros(max_sentence_len, dtype=np.int32)
        persona.append(pad_sentence)
        persona_lens.append(0)
    persona_lens = np.array(persona_lens, dtype=np.int32)
    persona = np.array(persona, dtype=np.int32)

    conversation = sentence_to_np(conversation, 
            max_conversation_words)

    response = sentence_to_np(response, max_sentence_len)

    return persona, conversation, response, persona_lens, \
        conversation_len, response_len

def get_batch_iterator(dataset, batch_size, max_sentence_len,
        max_conversation_words,
        max_persona_sentences):
    """ get an iterator over consecutive batches in the eval set

    note that we may miss up to batch_size-1 samples in the dataset
    """
    personas = []
    sentences = []
    responses = []
    persona_lens = []
    sentence_lens = []
    response_lens = []

    for sample in get_eval_sample_iterator(
            dataset, 
            max_sentence_len,
            max_conversation_words,
            max_persona_sentences):

        # break out the sample
        persona, conversation, response, persona_sentence_lens, \
            conversation_len, response_len = sample

        # add to lists
        personas.append(persona)
        sentences.append(conversation)
        responses.append(response)
        persona_lens.append(persona_sentence_lens)
        sentence_lens.append(conversation_len)
        response_lens.append(response_len)

        if len(personas) == batch_size:
            # convert everything to np arrays
            personas = np.array(personas)
            sentences = np.array(sentences)
            responses = np.array(responses)
            persona_lens = np.array(persona_lens)
            sentence_lens = np.array(sentence_lens)
            response_lens = np.array(response_lens)

            # convert to tensors
            personas = tf.constant(personas)
            sentences = tf.constant(sentences)
            responses = tf.constant(responses)
            persona_lens = tf.constant(persona_lens)
            sentence_lens = tf.constant(sentence_lens)
            response_lens = tf.constant(response_lens)

            yield personas, sentences, responses, \
                    persona_lens, sentence_lens, response_lens


def get_sample_iterator(dataset, max_sentence_len,
        max_conversation_words, max_persona_sentences, word2id):
    class ChatMarker:
        def __init__(self, chat, chat_index):
            self.free = []
            self.chat_index = chat_index
            for i in range(len(chat.chat)):
                self.free.append(i)
    
        def get_next_index(self):
            free_index = random.rand_int(0, len(self.free)-1)
            sample_index = self.free.pop(free_index)
            return sample_index
        
        def is_empty(self):
            return len(self.free) == 0
    
    # set up chat markers
    free_chats = []
    for i in range(len(dataset)):
        chat = dataset[i]
        current_marker = ChatMarker(chat)
        free_chats.append(current_marker)
    
    # start yielding samples
    while len(free_chats) > 0:
        # choose a chat
        free_index = random.rand_int(0, len(free_chats)-1)
        chat_marker = free_chats[free_index]

        chat_index = chat_marker.chat_index
        sample_index = chat_marker.get_next_index()

        # remove chat if there are no samples remaining
        if chat_index.is_empty() == True:
            del free_chats[free_index]

        yield get_sample(dataset, max_sentence_len, max_conversation_len,
                max_persona_sentences, word2id,
                chat_index, smaple_index)


#### CUT LINE

def get_training_batch(dataset, batch_size, max_sentence_len,
        max_conversation_len, max_conversation_words, 
        max_persona_sentences, word2id):
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
        persona_lens - List[List[int]] length of each persona sentence.
        sentence_lens - List[int] length of each conversation in sentences.
        response_lens - List[int] length of each sentence in responses.
    """
    # lists
    personas = []
    sentences = []
    responses = []
    persona_lens = []
    sentence_lens = []
    response_lens = []

    # fill lists with samples
    for i in range(batch_size):
        persona, conversation, response, persona_sentence_lens, \
                conversation_len, response_len = \
                get_full_sample(dataset, 
                        max_sentence_len, 
                        max_conversation_len,
                        max_conversation_words,
                        max_persona_sentences,
                        word2id)

        personas.append(persona)
        sentences.append(conversation)
        responses.append(response)
        persona_lens.append(persona_sentence_lens)
        sentence_lens.append(conversation_len)
        response_lens.append(response_len)

    # convert to np arrays
    personas = np.array(personas)
    sentences = np.array(sentences)
    responses = np.array(responses)
    persona_lens = np.array(persona_lens)
    sentence_lens = np.array(sentence_lens)
    response_lens = np.array(response_lens)

    # convert to tensors
    personas = tf.constant(personas)
    sentences = tf.constant(sentences)
    responses = tf.constant(responses)
    persona_lens = tf.constant(persona_lens)
    sentence_lens = tf.constant(sentence_lens)
    response_lens = tf.constant(response_lens)

    return personas, sentences, responses, persona_lens, sentence_lens, \
            response_lens

def get_eval_sample_iterator(dataset, max_sentence_len,
        max_conversation_len, max_conversation_words,
        max_persona_sentences):
    """ get an iterator over consecutive samples in the eval set.
    """
    for chat in dataset:
        # load up persona information
        persona = chat.your_persona
        persona_lens = []
        for persona_sentence in persona:
            persona_lens.append(len(persona_sentence))

        # convert persona to np arrays
        for i in range(len(persona)):
            sentence = persona[i]
            new_sentence = sentence_to_np(sentence, max_sentence_len)
            persona[i] = new_sentence

        # pad persona sentences
        while len(persona) < max_persona_sentences:
            pad_sentence = np.zeros(max_sentence_len, dtype=np.int32)
            persona.append(pad_sentence)
            persona_lens.append(0)
        persona_lens = np.array(persona_lens, dtype=np.int32)
        np_persona = np.array(persona, dtype=np.int32)
    
        # init the previous conversation
        conversation = []
        conversation_len = 0

        for i in range(0, len(chat.chat)):
            exchange = chat.chat[i]

            # add partner sentence to conversation
            for word in exchange[0]:
                conversation.append(word)
            conversation.append(0) # append id for "<pad>"
            conversation_len += len(exchange[0]) + 1

            # convert conversation
            np_conversation = sentence_to_np(conversation,
                    max_conversation_words)

            # get response
            response = exchange[1]
            response_len = len(exchange[1])

            # convert response
            np_response = sentence_to_np(response, max_sentence_len)

            # yield
            yield np_persona, np_conversation, np_response, persona_lens, \
                    conversation_len, response_len

            # add agent sentence to conversation
            for word in exchange[1]:
                conversation.append(word)
            conversation.append(0) # append id for "<pad>"
            conversation_len += len(exchange[1]) + 1
