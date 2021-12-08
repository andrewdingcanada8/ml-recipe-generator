import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.initializers import GlorotNormal, RandomNormal


class LSTM(tf.keras.Model):
  def __init__(self, embedding_dim, hidden_dim, vocab_sze, learning_rate, batch_sze):
    super(LSTM, self).__init__()
    
    self.optimizer = Adam(learning_rate=learning_rate)
    self.embedding_layer = Embedding(
      input_dim=vocab_sze,
      output_dim=embedding_dim,
      embeddings_initializer=RandomNormal(), #verify randomnormal
      batch_input_shape=[batch_sze,]
    )
    self.lstm_layer = tf.keras.layers.LSTM(
      units=hidden_dim,
      return_sequences=True,
      stateful=True,
      recurrent_initializer=GlorotNormal()
    )
    self.dense_layer = Dense(vocab_sze, activation='sigmoid')
  
  def call(self, inputs):
    embeddings = self.embedding_layer(inputs)
    output = self.lstm_layer(embeddings)
    return self.dense_layer(output)
  
  def loss(self, labels, preds):
    entropy = tf.keras.losses.sparse_categorical_crossentropy(
      y_true=labels,
      y_pred=preds,
      from_logits=True
    )
    return entropy