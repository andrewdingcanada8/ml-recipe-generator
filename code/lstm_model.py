import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.initializers import GlorotNormal, RandomNormal


class LSTM():
  def __init__(self, embedding_dim, hidden_dim, vocab_sze, learning_rate):
    self.optimizer = Adam(learning_rate=learning_rate)
    self.sequential_layer = Sequential([
      Embedding(
        vocab_sze,
        embedding_dim,
        embeddings_initializer=RandomNormal() #verify randomnormal
      ),
      LSTM(
        embedding_dim,
        hidden_dim,
        vocab_sze,
        return_sequences=True,
        stateful=True,
        recurrent_initializer=GlorotNormal()
      ),
      Dense(vocab_sze)
    ])
  
  def call(self, inputs):
    return self.sequential_layer(inputs)
  
  def loss(self, labels, logits):
    entropy = tf.keras.losses.sparse_categorical_crossentropy(
      y_true=labels,
      y_pred=logits,
      from_logits=True
    )
    return entropy