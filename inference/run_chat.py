
import numpy as np
import logging
import sys

from util.data_util import get_personas
from util.data_util import convert_sentence_to_id
from util.data_util import sentence_to_np


def run_inference(model, data, word2id, num_conversations = -1):
    while num_conversations > 0 or num_conversations == -1:
        run_conversation(model, data, word2id)

        if num_conversations > 0:
            num_conversations -= 1

def run_conversation(model, data, word2id):
    # TODO modify this so it doesn't rely on model class variables

    # get personas from data
    p1_tuple, p2_tuple = get_personas(data, model.max_sentence_len,
            model.max_conversation_len, model.max_persona_sentences)

    p1 = p1_tuple[0]
    p1_lens = p1_tuple[1]
    p2 = p2_tuple[0]
    p2_lens = p2_tuple[1]

    # run conversation
    #print("__SILENCE__")

    # build start of conversation
    string_conversation = "__SILENCE__ <pad>"
    id_conversation = []
    id_conversation = convert_sentence_to_id(string_conversation,
            word2id)
    conversation_len = len(id_conversation)
    id_conversation = sentence_to_np(id_conversation, 
            model.max_sentence_len * model.max_conversation_len)

    # reshape everything to batch size 1
    p1 = np.reshape(p1, (1, model.max_persona_sentences,
        model.max_sentence_len))
    p2 = np.reshape(p2, (1, model.max_persona_sentences,
        model.max_sentence_len))

    id_conversation = np.reshape(id_conversation,
            (1, model.max_sentence_len * model.max_conversation_len))

    p1_lens = np.reshape(p1_lens, (1, model.max_persona_sentences))
    p2_lens = np.reshape(p2_lens, (1, model.max_persona_sentences))

    conversation_len = np.array(conversation_len)
    conversation_len = np.reshape(conversation_len, (1, 1))


    for i in range(model.max_conversation_len):
        # get first agent response
        feed_dict = {
                model.persona_sentences: p1,
                model.context_sentences: id_conversation,
                model.persona_sentence_lens: p1_lens,
                model.context_sentence_lens: conversation_len
            }

        a1_string_response, a1_id_response = model.sess.run(
                [model.text_output, model.output_predictions],
                feed_dict = feed_dict)

        # add to conversation
        print(a1_string_response)
        id_conversation += a1_id_response
        conversation_len[0, 0] += len(a1_response_id)

        # get second agent response
        feed_dict = {
                model.persona_sentences: p2,
                model.context_sentences: id_conversation,
                model.persona_sentence_lens: p2_lens,
                model.context_sentence_lens: np.array(
                    conversation_len)
                }

        a2_string_response, a2_id_response = model.sess.run(
                [model.output_string, model.output_predictions],
                feed_dict = feed_dict)

        # add to conversation
        print(a2_string_response)
        id_conversation += a2_id_response
        conversation_len[0, 0] += len(a2_response_id)

