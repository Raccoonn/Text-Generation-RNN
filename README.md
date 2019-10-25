# Text-Generation-RNN

This all started when I saw the Tensor Flow tutorial for Text Generation using a Recurrent Neural Network.  After doing the tutorial using the Shakespeare data set I thought it would be fun build an RNN on a data set of Twitch Chat messages.  I abstracted the Tensor Flow tutorial into a command line input where you can create or load a model and use it to generate text.  The command line input asks for a data set and a checkpoints folder.  Use any data set and train the model to create a checkpoints folder that can be into the network later.


Also included in this repository is processChat.py which contains useful functions for "cleaning" raw chat data by removing non-standard characters.  processChat.py also has a command line input to select a chat.log file to "clean" and create a clean chat.txt file.  Explore this script by using the provided #example_chat.log file.


For creating your own data set, check out my Twitch Chat Bot repository.  Using the bot will allow you to connect to live Twitch chat irc channels and write the data to file for use in the RNN.


In the end we're using gibberish to generate gibberish but I had alot of fun putting this together.  As I mentioned this RNN is just an abstraction of the Tensor Flow turotial and thus can be used for any input set with compatible characters.
