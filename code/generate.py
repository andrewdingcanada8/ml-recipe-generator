import numpy as np
import tensorflow as tf


def generate_text(model, tokenizer, start_string, num_words=201, temp=0.5):
    """
        generate_text(model, tokenizer, start_string, num_words, temp): generates the text
        uses the model and staring words to create a sequence, then uses tokenizer to convert back to text
            num_words is the hard limit on recipe length and 
            temp decides how 'risky'/experimental the generation is
    """
    # padded_start_string = STOP_WORD
    input_indices = np.array(tokenizer.texts_to_sequences([start_string]))
    # Empty string to store our results.
    text_generated = []

    # Here batch size == 1.
    model.reset_states()
    for _ in range(num_words):
        probs = model.call(input_indices)

        # remove the batch dimension
        probs = tf.squeeze(probs, 0)

        # Using a categorical distribution to predict the character returned by the model.
        probs = probs / temp
        predicted_id = tf.random.categorical(
            logits=probs,
            num_samples=1
        )[-1, 0].numpy()

        # get the word from the predicted words, and use them as input for next call
        input_indices = tf.expand_dims([predicted_id], 0)
        next_word = tokenizer.sequences_to_texts(input_indices.numpy())[0]

        if (next_word == "<stop>"):
            break

        text_generated.append(next_word)

    return (start_string + " " + ' '.join(text_generated))
