import pickle
import json
import tensorflow as tf
import math
import numpy as np


def get_data(data_path, tokenizer_path, batch_sze, percent=1.0):
    with open(data_path, 'rb') as file:
        data = pickle.load(file)

    # slice data into smaller set for debugging
    assert isinstance(percent, float) and percent <= 1.0
    data = data[:math.floor(len(data)*percent)].astype('uint64')

    # create a dataset
    dataset = tf.data.Dataset.from_tensor_slices(data)
    with open(tokenizer_path, 'rb') as file:
        tokenizer = pickle.load(file)

    print(tokenizer.sequences_to_texts([data[1]]))
    inputs = dataset.map(lambda input: (input[:-1], input[1:]))
    word_counts = json.loads(tokenizer.get_config()['word_index'])
    assert isinstance(word_counts, dict)
    vocab_sze = len(word_counts)

    # Shuffle & batch dataset
    # repeat is used so the dataset can be trained on repeatedly through
    # multiple epochs
    inputs = inputs.shuffle(10*batch_sze).batch(batch_sze, drop_remainder=True)
    return inputs, tokenizer, vocab_sze, data.shape[1]
