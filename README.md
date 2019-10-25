# Text-Generation-RNN

This all started when I saw the Tensor Flow tutorial for Text Generation using a Recurrent Neural Network.  After doing the tutorial using the Shakespeare data set I thought it would be fun build an RNN on a data set of Twitch Chat messages.  I abstracted the Tensor Flow tutorial into a command line input where you can create or load a model and use it to generate text.  For creating a model a data set will be required for input, for loading a model a checkpoints folder will be loaded.


Also included in this repository is processChat.py which contains useful functions for "cleaning" raw chat data by removing non-standard characters.  cleanChat.py is a command line input to select a chat.log file to "clean" and create a clean .txt file.  Explore these scripts by using the provided #example_chat.log file.


For creating your own data set, check out my Twitch Chat Bot repository.  Using the bot will allow you to connect to live Twitch chat irc channels and write the data to file for use in the RNN.
