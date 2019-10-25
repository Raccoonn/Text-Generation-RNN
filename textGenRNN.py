########################################################################################
#
#    CURRENT STEPS:
#
#        - Create textGenRNN object  (currently assigns all problem specific data
#          for a cleanChat.txt file specified in path_to_file)
#
#        - Use rnn.new_model() to construct a new model with the larger, 64, BATCH_SIZE
#
#        - Once a new model has been constructed use rnn.train() to train the model.
#          Currently EPOCHS is specified within the train() function.
#
#        - After the model has trained use rnn.load_model() to load the model from the`
#          last training checkpoint.  Have to use load_model() to change BATCH_SIZE = 1.
#
#        - Now that a model has been created and trained the checkpoints can be loaded
#          for a new instance of the model.
#
#
########################################################################################


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import os
import time
import progressbar




class textGenRNN:
    def __init__(self, path_to_file, checkpoint_dir):
        '''
        Setup data
            - Reads in .txt file containing data and creates librarys for char2idx
              and idx2char and formats data.
            - Defines BUFFER_SIZE and BATCH_SIZE as well as embedding and rnn_units
              for the network.
            - Setups up directory and file name for checkpoint data.
        '''
        self.text = open(path_to_file, 'r').read()
        self.vocab = sorted(set(self.text))

        self.char2idx = {u: i for i, u in enumerate(self.vocab)}

        self.idx2char = np.array(self.vocab)
        text_as_int = np.array([self.char2idx[c] for c in self.text])


        seq_length = 100
        char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
        sequences = char_dataset.batch(seq_length+1, drop_remainder=True)


        def split_input_target(chunk):
            input_text = chunk[:-1]
            target_text = chunk[1:]
            return input_text, target_text


        self.dataset = sequences.map(split_input_target)

        self.BATCH_SIZE = 64
        self.BUFFER_SIZE = 10000

        self.dataset = self.dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder=True)

        self.vocab_size = len(self.vocab)
        self.embedding_dim = 256
        self.rnn_units = 1024
        
        self.checkpoint_dir = './' + checkpoint_dir
        


    def build_model(self, vocab_size, embedding_dim, rnn_units, batch_size):
        '''
        Function to build a model with the specified attributes.  Made this a separate
        method so it can be called in other methods.
        '''
        self.model = tf.keras.Sequential([
                
                    tf.keras.layers.Embedding(vocab_size,
                                              embedding_dim,
                                              batch_input_shape=[batch_size, None]
                                              ),

                    tf.keras.layers.GRU(rnn_units,
                                        return_sequences=True,
                                        stateful=True,
                                        recurrent_initializer='glorot_uniform'
                                        ),

                    tf.keras.layers.Dense(vocab_size)
                    
                    ])



    def new_model(self):
        '''
        Calls build_model to create a new model for the network.
        '''
        self.build_model(self.vocab_size, self.embedding_dim, self.rnn_units, batch_size=self.BATCH_SIZE)
        self.model.summary()



    def load_model(self):
        '''
        Load model from checkpoint directory with BATCH_SIZE = 1
        '''
        tf.train.latest_checkpoint(self.checkpoint_dir)
        self.build_model(self.vocab_size, self.embedding_dim, self.rnn_units, batch_size=1)
        self.model.load_weights(tf.train.latest_checkpoint(self.checkpoint_dir))
        self.model.build(tf.TensorShape([1, None]))
        self.model.summary()




    def train(self, EPOCHS=10):
        '''
        Train model, can specify EPOCHS
        '''
        def loss(labels, logits):
            return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

        self.model.compile(optimizer='adam', loss=loss)

        checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt_{epoch}")

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix, save_weights_only=True)

        self.model.fit(self.dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])



    def generate_text(self, start_string, num_generate=1000, temperature=1.0):
        '''
        Given a string of form u'blah blah blah' this will return the generated text
        up to num_generate.
        '''
        widgets = [progressbar.Percentage(), progressbar.Bar()]
        bar = progressbar.ProgressBar(widgets=widgets, max_value=num_generate).start()

        # Converting our start string to numbers (vectorizing)
        input_eval = [self.char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # Empty string to store our results
        text_generated = []

        # Low temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        # Experiment to find the best setting.
        # temperature = 1.0

        # Here batch size == 1
        self.model.reset_states()
        for i in range(num_generate):
            predictions = self.model(input_eval)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            # using a categorical distribution to predict the word returned by the model
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(
                predictions, num_samples=1)[-1, 0].numpy()

            # We pass the predicted word as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(self.idx2char[predicted_id])

            bar.update(i+1)
        bar.finish()

        print('\n\nBelow is %d characters of generated text:\n' % num_generate)
        return (start_string + ''.join(text_generated))

