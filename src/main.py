import pandas as pd
import numpy as np
import string
import re
import nltk
from random import randint
ps = nltk.PorterStemmer()

def clean_text(text):
    text = text.replace('--', ' ')
    tokens = text.split()
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    return tokens


def gen_data(lyrics_data_sample,len):
    try:
        print("Slicing and cleaning data")
        lyrics_data = lyrics_data[pd.notnull(lyrics_data['lyrics'])]
        lyrics_data_sample = lyrics_data[0:len]
        lyrics_data_sample['lyrics'] = lyrics_data_sample['lyrics'].apply(lambda x: clean_text(x))
        return lyrics_data_sample
    except:
        print("length provided exceeded actual rows")

def main():
    lyrics_data = pd.read_csv("../data/lyrics.csv")
    data = gen_data(lyrics_data,10000)
    tokens = list()
    for row in data['lyrics']:
        tokens += row

    length = 51
    lines = list()
    for i in range(0,len(tokens)-len(tokens)%length,length):
        seq = tokens[i:i+length]
        line = ' '.join(seq)
        lines.append(line)
    print('Total Sequences: %d' % len(lines))

    #tokensize the words to integers
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    sequences = tokenizer.texts_to_sequences(lines)

    sequences = np.array(sequences)
    vocab_size = len(tokenizer.word_index) + 1

    X, y = sequences[:,:-1], sequences[:,-1]
    y = to_categorical(y, num_classes=vocab_size)

    # create model
    embedding_size = 50
    batch_size = 128
    epochs = 50
    nlg_model  = NlgModel(X,y,vocab_size,embedding_size,batch_size,epochs)

    #printing and save model summary
    nlg_model.get_model_summary();

    #train the model
    nlg_model.train()

    #save model
    nlg_model.save_model()

    #generate lyrics sequences given some seed_text
    generated = nlg_model.generate_seq(tokenizer, 50, lines[randint(0,len(lines))], 50)
    print("Given the seed text as ---> ",lines[randint(0,len(lines))])
    print("Text generated was this ---> ",generated)


if __name__== "__main__" :
    main()
