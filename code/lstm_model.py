import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding
from tensorflow.keras.initializers import GlorotNormal, RandomNormal


class Custom_LSTM(tf.keras.Model):
    def __init__(self, embedding_dim, hidden_dim, vocab_sze, learning_rate, batch_sze, recipe_len):
        super(Custom_LSTM, self).__init__()

        # read in the values and set in the model
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_sze = vocab_sze
        self.learning_rate = learning_rate
        self.batch_sze = batch_sze
        self.recipe_len = recipe_len

        # inti the optimizer
        self.optimizer = Adam(learning_rate=learning_rate)

        # the embedding layer
        self.embedding_layer = Embedding(
            input_dim=recipe_len,
            output_dim=embedding_dim,
            embeddings_initializer=RandomNormal(),  # verify randomnormal
            batch_input_shape=[batch_sze, ]
        )

        # the LSTM layer
        self.custom_lstm_layer = tf.keras.layers.LSTM(
            units=hidden_dim,
            return_sequences=True,
            stateful=True,
            recurrent_initializer=GlorotNormal()
        )

        # dense layer at the end
        # note: no activation because we use from_logits in loss
        self.dense_layer = Dense(vocab_sze)

    def call(self, inputs):
        embeddings = self.embedding_layer(inputs)
        output = self.custom_lstm_layer(embeddings)
        return self.dense_layer(output)

    def loss(self, labels, preds):
        entropy = tf.keras.losses.sparse_categorical_crossentropy(
            y_true=labels,
            y_pred=preds,
            from_logits=True
        )

        return entropy
