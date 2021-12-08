import argparse
import json
import sys
import os
import pickle
import tensorflow as tf
from lstm_model import LSTM
from preprocess import get_data

# def test(model, test_inputs, test_labels):
#     """
#     Runs through one epoch - all testing examples

#     :param model: the trained model to use for prediction
#     :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
#     :param test_labels: train labels (all labels for testing) of shape (num_labels,)
#     :returns: perplexity of the test set
#     """
#     total_batch_size = model.batch_size*model.window_size    
#     batch_loss = []
    
#     margin = -(len(test_inputs)%(model.batch_size*model.window_size))
#     test_inputs = test_inputs[:margin]
#     test_labels = test_labels[:margin]
#     input_size = len(test_inputs)
#     num_batches = input_size//total_batch_size
#     test_inputs = tf.reshape(test_inputs, (num_batches, model.batch_size, model.window_size))
#     test_labels = tf.reshape(test_labels, (num_batches, model.batch_size, model.window_size))
    
#     for index in range(0, len(test_inputs)):
#         inputs_batch = test_inputs[index]
#         labels_batch = test_labels[index]
#         probs, state = model.call(inputs_batch, None)
#         batch_loss.append(model.loss(probs, labels_batch))
#     return tf.exp((np.array(batch_loss).mean()))

def train(model, inputs, epoch):
  prog = tf.keras.utils.Progbar(len(inputs))
  total_loss = 0
  for train_x, train_y in inputs:
    # print(train_x.shape, train_y.shape)
    with tf.GradientTape() as tape:
      preds = model.call(train_x)
      loss = model.loss(train_y, preds)
    gradients = tape.gradient(loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    total_loss += loss
    prog.add(1,[('epoch', int(epoch)), ('loss', loss)])
  return total_loss
  
def parseArguments():
  parser = argparse.ArgumentParser()
  parser.add_argument("--embedding_dim", type=int, default=16) #256
  parser.add_argument("--hidden_dim", type=int, default=32) #1024
  parser.add_argument("--epochs", type=int, default=3)
  parser.add_argument("--batch_size", type=int, default=64)
  parser.add_argument("--learning_rate", type=float, default=1e-3)
  parser.add_argument("--percent", type=float, default=1.0)
  args = parser.parse_args()
  return args

# def train(model, input):
#   for input_example_batch, target_example_batch in input.take(1):
#     preds = model(input_example_batch)
    
def main(args):
    # initialize parameters
    EMBEDDING_DIM = args.embedding_dim
    HIDDEN_DIM = args.hidden_dim
    EPOCHS = args.epochs
    BATCH_SZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    PERCENT = args.percent
    
    # import data from preprocessing
    inputs, tokenizer, vocab_sze = get_data(
      'data/saved_data/data_padded.pickle', 
      'data/saved_data/tokenizer.pickle',
      BATCH_SZE,
      PERCENT
      )
    
    # STEPS_PER_EPOCH = #total // batch size
    STEPS_PER_EPOCH = 1500 #temp
    VOCAB_SZE = vocab_sze
    
    # initialize model
    model = LSTM(EMBEDDING_DIM, HIDDEN_DIM, VOCAB_SZE, LEARNING_RATE, BATCH_SZE)
    
    losses = []
    for epoch in range(1, EPOCHS+1):
      total_loss = train(model, inputs, epoch)
      losses.append(total_loss)
    


if __name__ == '__main__':
    args = parseArguments()
    main(args)
