import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.initializers import GlorotNormal, RandomNormal


class LSTM():
  def __init__(self, learning_rate):
    self.optimizer = Adam(learning_rate=learning)
    self.sequential_layer = Sequential()
  def loss(self, labels, logits):
    entropy = tf.keras.losses.sparse_categorical_crossentropy(
      y_true=labels,
      y_pred=logits,
      from_logits=True
    )
    return entropy
  def create_model(self, embedding_dim, hidden_dim, vocab_sze):
    self.model.add(Embedding(
      vocab_sze,
      embedding_dim,
      embeddings_initializer=RandomNormal() #verify randomnormal
    ))
    # verify: the nb had return_sequences = true, not sure why (also in rnn assignment)
    self.model.add(LSTM(
      embedding_dim,
      hidden_dim,
      vocab_sze,
      return_sequences=True,
      stateful=True,
      recurrent_initializer=GlorotNormal()
    ))
    self.model.add(Dense(vocab_sze))
    self.model.compile(
      optimizer=self.optimizer,
      loss=self.loss
    )
    return self.model
  