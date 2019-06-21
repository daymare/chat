
import numpy as np
import logging
import sys

from util.train_util import get_personas
from util.train_util import sentence_to_np
from util.data_util import convert_sentence_to_id
from util.data_util import convert_sentence_from_id

from model import Model

import tensorflow as tf


def run_inference(data, config, word2id, word2vec, id2word, num_conversations = -1, user_chat=False):
    # load models
    agent1 = Model(config, word2vec, id2word, word2id)
    agent2 = Model(config, word2vec, id2word, word2id)
    agent1.load(config.checkpoint_dir)
    agent2.load(config.checkpoint_dir)

    while num_conversations > 0 or num_conversations == -1:
        run_conversation(agent1, agent2, data, word2id, 
                config, user_chat)

        if num_conversations > 0:
            num_conversations -= 1

def run_conversation(agent1, agent2, data, word2id, 
        config, user_chat):
    # TODO implement user chat
    # get personas from data
    persona1, persona2 = get_personas(data, config.max_sentence_len,
            config.max_conversation_len, config.max_persona_len, config.max_conversation_words)

    # perform first exchange
    conversation = "__SILENCE__ <pad>"
    id_conversation = convert_sentence_to_id(conversation, word2id)
    id_conversation = tf.expand_dims(id_conversation, 0)

    agent1_str, agent1_ids = agent1(id_conversation, persona1, reset=True)
    agent1_ids = tf.expand_dims(agent1_ids, 0)

    print(agent1_str)

    agent2_str, agent2_ids = agent2(agent1_ids, persona2, reset=True)
    agent2_ids = tf.expand_dims(agent2_ids, 0)

    print(agent2_str)

    # continue the conversation
    for t in range(config.max_conversation_len):
        # perform next exchange
        agent1_str, agent1_ids = agent1(agent2_ids)
        agent1_ids = tf.expand_dims(agent1_ids, 0)

        print(agent1_str)

        agent2_str, agent2_ids = agent2(agent1_ids)
        agent2_ids = tf.expand_dims(agent2_ids, 0)

        print(agent2_str)





    



