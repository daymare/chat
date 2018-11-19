


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
