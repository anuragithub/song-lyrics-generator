from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.utils import plot_model




class NlgModel(object):
    def __init__(self,X,y,vocab_size,embedding_size,batch_size,epochs):

        self.input_X = X
        self.input_y = y
        self.vocab_size = vocab_size
        self.embed_size = embedding_size
        self.batch_size = batch_size
        self.epochs = epochs
        #self.lstm_layers = n_lstm #we can tweak to use it later

        # model creation
        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size, self.embed_size, input_length=self.input_X.shape[1]))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(LSTM(100))
        self.model.add(Dense(100, activation='relu'))
        self.model.add(Dense(self.vocab_size, activation='softmax'))

    # model summary
    def get_model_summary():
        print("Printing and saving model summary")
        print(self.modelsummary())
        plot_model(self.model, to_file='model.png')


    ##train model
    def train():
        # compile model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # fit model
        print("Training the model now...")
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs)

    ## save models
    def save_model():
        self.model.save_weights("model.h5")
        print("Saved model to disk")
        
    ## generate return_sequences
    def generate_seq(tokenizer, seq_length, seed_text, n_words):
        printing("generating sequence now...")
    	result = list()
    	in_text = seed_text
    	# generate a fixed number of words
    	for _ in range(n_words):
    		# encode the text as integer
    		encoded = tokenizer.texts_to_sequences([in_text])[0]
    		# truncate sequences to a fixed length
    		encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
    		# predict probabilities for each word
    		yhat = self.model.predict_classes(encoded, verbose=0)
    		# map predicted word index to word
    		out_word = ''
    		for word, index in tokenizer.word_index.items():
    			if index == yhat:
    				out_word = word
    				break
    		# append to input
    		in_text += ' ' + out_word
    		result.append(out_word)
    	return ' '.join(result)
