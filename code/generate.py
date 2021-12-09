import numpy as np
import tensorflow as tf

def generate_text(model, tokenizer, start_string, num_words=1000, temp=0.5):
  # padded_start_string = STOP_WORD
  input_indices = np.array(tokenizer.texts_to_sequences([start_string]))
  print(f'input_indices shape: {input_indices.shape}')
  # Empty string to store our results.
  text_generated = []

  # Here batch size == 1.
  model.reset_states()
  for _ in range(num_words):
    logits = model.call(input_indices)
    # remove the batch dimension
    logits = tf.squeeze(logits, 0)

    # Using a categorical distribution to predict the character returned by the model.
    logits = logits / temp
    predicted_id = tf.random.categorical(
      logits=logits,
      num_samples=1
    )[-1,0].numpy()

    # We pass the predicted character as the next input to the model
    # along with the previous hidden state.
    input_indices = tf.expand_dims([predicted_id], 0)
    # np.roll(input_indices, -1)
    # input_indices[-1] = predicted_id
    
    next_character = tokenizer.sequences_to_texts(input_indices.numpy())[0]

    text_generated.append(next_character)

  return (start_string + ''.join(text_generated))