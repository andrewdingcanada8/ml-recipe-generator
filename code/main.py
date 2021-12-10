import argparse
import os
import pickle
from re import split

import tensorflow as tf
from generate import generate_text

from lstm_model import Custom_LSTM
from preprocess import get_data
from generate import generate_text

import numpy as np


def train(model, inputs, epoch):
    """
        train(model, inputs, epoch): trains the model based on the inputs
        we use similar methods taught to us in class - using model.call and then model.loss, with GradientTape 
    """
    prog = tf.keras.utils.Progbar(len(inputs))
    total_loss = 0

    i = 0
    for train_x, train_y in inputs:

        with tf.GradientTape() as tape:
            preds = model.call(train_x)
            loss = model.loss(train_y, preds)

        # debug for if loss goes to NaN
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
    """
        parseArguments(): parses command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dim", type=int, default=256)  # 256
    parser.add_argument("--hidden_dim", type=int, default=1024)  # 1024
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--percent", type=float, default=1.0)

    # choose to train or test
    parser.add_argument('--test_only', dest='test_only', action='store_true')
    parser.add_argument('--train', dest='test_only',
                        action='store_false')
    parser.set_defaults(test_only=True)

    args = parser.parse_args()

    print(args)
    return args


def main(args):
    """
        main(args): main function
        loads data, inits the model and trains or loads weights based on args
        runs a repl if run under test mode which takes in starting input and returns a generated recipe
    """

    # initialize parameters
    embedding_dim = args.embedding_dim
    hidden_dim = args.hidden_dim
    epochs = args.epochs
    batch_sze = args.batch_size
    learning_rate = args.learning_rate
    percent = args.percent
    test_only = args.test_only

    # import data from preprocessing
    inputs, tokenizer, vocab_sze, recipe_len = get_data(
        'data/saved_data/data_padded.pickle',
        'data/saved_data/tokenizer.pickle',
        batch_sze,
        percent
    )

    # initialize model
    model = Custom_LSTM(embedding_dim, hidden_dim,
                        vocab_sze, learning_rate, batch_sze, recipe_len)

    input_shape = (batch_sze, recipe_len)
    model.build(input_shape)
    model.compute_output_shape(input_shape)

    model_weight_path = 'code/models/lstm'

    # only train if we want to
    if not test_only:
        if (os.path.exists(model_weight_path)):
            while (True):
                print("\nmodel alread exists, do you want to continue training? (y/n)")
                i = input()

                # load the weights and continue training the model
                if (i == "y"):
                    model.load_weights(model_weight_path).expect_partial()
                    break
                else:
                    break

        # train the model
        losses = []
        for epoch in range(1, epochs+1):
            total_loss = train(model, inputs, epoch)
            losses.append(total_loss)

        # save the model after training
        model.save_weights(model_weight_path,
                           save_format="tf", overwrite=True)

    else:
        # load the model weights
        model.load_weights(model_weight_path).expect_partial()

    print("\n\n\n\nwelcome to Salad Party recipe generator. we provide first class recipes developed by our friendly LSTM model")

    # repl for testing
    while True:
        print("\n\nplease enter starting words (ingredients)\n")
        i = input()

        assert(i and len(i) > 0 and len(i.split(" ")) > 0)
        if (i == "exit"):
            return

        split_i = i.split(" ")

        # create new model and transfer weights for the prediction
        new_model = Custom_LSTM(embedding_dim, hidden_dim,
                                vocab_sze, learning_rate, 1, recipe_len)

        new_model.build((1, len(split_i)))
        new_model.set_weights(model.get_weights())

        # generate the text and remove stop words
        combo = generate_text(new_model, tokenizer, i, num_words=recipe_len)
        print("\n\nyour special recipe is ready!\n")
        print(combo.replace("<stop>", ""))


if __name__ == '__main__':
    args = parseArguments()
    main(args)
