import argparse
import json
import os
import pickle
from re import L
import sys

import tensorflow as tf
from generate import generate_text

from lstm_model import Custom_LSTM
from preprocess import get_data
from generate import generate_text

import numpy as np

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

    i = 0
    for train_x, train_y in inputs:
        # print(train_x.shape, train_y.shape)

        with tf.GradientTape() as tape:
            preds = model.call(train_x)
            loss = model.loss(train_y, preds)

        if (np.any(np.isnan(loss))):
            with open("error_file.pickle", 'wb+') as file:
                pickle.dump(train_x, file)
            print(i)
            break

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(
            zip(gradients, model.trainable_variables))

        total_loss += loss
        prog.add(1, [('epoch', int(epoch)), ('loss', loss)])
        i += 1
    return total_loss


def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dim", type=int, default=256)  # 256
    parser.add_argument("--hidden_dim", type=int, default=1024)  # 1024
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--percent", type=float, default=1.0)

    parser.add_argument('--test_only', dest='test_only', action='store_true')
    parser.add_argument('--train', dest='test_only',
                        action='store_false')
    parser.set_defaults(test_only=True)

    args = parser.parse_args()

    print(args)
    return args


def main(args):
    # initialize parameters
    EMBEDDING_DIM = args.embedding_dim
    HIDDEN_DIM = args.hidden_dim
    EPOCHS = args.epochs
    BATCH_SZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    PERCENT = args.percent
    TEST_ONLY = args.test_only

    # import data from preprocessing
    inputs, tokenizer, vocab_sze, recipe_len = get_data(
        'data/saved_data/data_padded.pickle',
        'data/saved_data/tokenizer.pickle',
        BATCH_SZE,
        PERCENT
    )

    # print(np.sum(inputs))
    # return

    # STEPS_PER_EPOCH = #total // batch size
    STEPS_PER_EPOCH = 1500  # temp

    # initialize model
    model = Custom_LSTM(EMBEDDING_DIM, HIDDEN_DIM,
                        vocab_sze, LEARNING_RATE, BATCH_SZE, recipe_len)
    input_shape = (BATCH_SZE, recipe_len)
    model.build(input_shape)
    model.compute_output_shape(input_shape)
    # print(model.get_weights())

    model_weight_path = 'code/models/lstm'

    # only train if we want to
    if not TEST_ONLY:
        if (os.path.exists(model_weight_path)):
            while (True):
                print("model exists, are you sure you wanna train again (y/n)")
                i = input()

                # train it
                if (i == "y"):
                    losses = []
                    for epoch in range(1, EPOCHS+1):
                        total_loss = train(model, inputs, epoch)
                        losses.append(total_loss)

                    model.save(model_weight_path, save_format="tf")
                    break

                # dont train
                elif (i == "n"):
                    tf.keras.models.load_model(model_weight_path)
                    break
        else:
            losses = []
            for epoch in range(1, EPOCHS+1):
                total_loss = train(model, inputs, epoch)
                losses.append(total_loss)

            model.save(model_weight_path, save_format="tf")

    else:
        tf.keras.models.load_model(model_weight_path)

    new_model = Custom_LSTM(EMBEDDING_DIM, HIDDEN_DIM,
                            vocab_sze, LEARNING_RATE, 1, recipe_len)
    new_model.build((1, 2))
    new_model.set_weights(model.get_weights())

    combo = generate_text(new_model, tokenizer,
                          "mushroom and strawberries", num_words=recipe_len)
    print(combo)


if __name__ == '__main__':

    args = parseArguments()
    main(args)
