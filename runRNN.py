########################################################################################
#
# Script to execute the Text Generation Recurrent Neural Network
#
#     - Specify full address for path_to_file
#
#     - Specify name for checkpoint_dir, folder will be created in the
#       working directory.
#
#     - build_model to create a new model to be trained
#
#     - load_model to load model weights from the specified checkpoint_dir
#
#     - train_model to train, can also specify EPOCHS
#
#     - generate_text to pass a string to the network, can also specify output
#       length and temperature (low temp is more conservative).
#
#
# Currently all functions work for the single RNN object as the path_to_file
# and checkpoint_dir are both set in __init__
#
########################################################################################


from textGenRNN import textGenRNN
import os



if __name__ == '__main__':

    input('\n\n\nText Generation Recurrent Neural Network \n\n \
           Abilities:  - Create and train a new neural network on the data \n \
                         provided.\n\n \
                       - Once model is trained or if a checkpoint_dir is \n \
                         saved use Load Model to "activate" the model. \n\n \
                       - After loading use Generate Text to input starting \n \
                         strings, adjust sequence output and temperature. \n\n\n \
           Press Enter to begin.\n\n')

    # Get path for dataset
    while True:
        path_to_file = input('\n\nInput filename for data set: ')
        if os.path.exists(path_to_file) == True:
            break  
        else:
            print('\n' + path_to_file + ' file not found.\n')

    # Specify folder for checkpoint storage
    checkpoint_dir = input('\nSpecify folder name for checkpoints: ')

    # Initialize RNN
    rnn = textGenRNN(path_to_file, checkpoint_dir)

    # Choose process
    choices = ['New Model', 'Load Model', 'Quit']
    while True:
        while True:
            pick = input('\n\nNew Model, Load Model, or Quit: ')
            if pick in choices:
                break
            else:
                print('\nInvalid Input\n')


        # Quit loop and end script, checkpoint data folder is left
        if pick == 'Quit':
            break

    
        # Build new model and begin training
        elif pick == 'New Model':
            rnn.new_model()
            print('\nNew model created')
            EPOCHS = int(input('\nSpecify training EPOCHS: '))

            # Begin training
            input('\nPress Enter to begin training.')
            rnn.train(EPOCHS)
        

        # Load model after training or from previous checkpoint folder
        elif pick == 'Load Model':
            rnn.load_model()
            print('\nModel loaded succesfully\n')

            # Run a loop to play with text generation
            while True:
                start = input('\nInput a starting string: ')
                size = int(input('\nSpecify output size: '))
                temp = float(input('\nSpecify temperature: '))
                print('\nGenerating text:')
                print(rnn.generate_text(start, size, temp) + '\n')

                next = input('\n\nGenerate more text (Y/N): ')
                if next in ['Y', 'y']:
                    pass
                else:
                    break
