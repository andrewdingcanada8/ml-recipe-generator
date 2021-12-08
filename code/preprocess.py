import pickle
import json
import tensorflow as tf

def get_data(data_path, tokenizer_path, batch_sze):
    with open(data_path, 'rb') as file:
      data = pickle.load(file)
    dataset = tf.data.Dataset.from_tensor_slices(data)
    with open(tokenizer_path, 'rb') as file:
      tokenizer = pickle.load(file)
    inputs = dataset.map(lambda input: (input[:-1], input[1:]))
    word_counts = json.loads(tokenizer.get_config()['word_counts'])
    assert isinstance(word_counts, dict)
    vocab_sze = len(word_counts)
    
    # Shuffle & batch dataset
    # repeat is used so the dataset can be trained on repeatedly through
    # multiple epochs
    inputs = inputs.shuffle(10*batch_sze).batch(batch_sze, drop_remainder=True).repeat()
    return inputs, tokenizer, vocab_sze