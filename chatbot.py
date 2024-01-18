
#using Tensorflow
import os
import json
import random
import pickle

from typing import Union

import nltk
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, InputLayer
from tensorflow.keras.optimizers import Adam, Optimizer


class Assistant:

    def __init__(self, intents_data: Union[str, os.PathLike, dict], method_mappings: dict = {}, hidden_layers: list = None, model_name: str = "basic_model") -> None:

        #nltk.download('punkt', quiet=True)
        #nltk.download('wordnet', quiet=True)

        if isinstance(intents_data, dict):
            self.intents_data = intents_data
        else:
            if os.path.exists(intents_data):
                with open(intents_data, "r") as f:
                    self.intents_data = json.load(f)
            else:
                raise FileNotFoundError

        self.method_mappings = method_mappings
        self.model = None
        self.hidden_layers = hidden_layers
        self.model_name = model_name
        self.history = None

        self.lemmatizer = nltk.stem.WordNetLemmatizer()

        self.words = []
        self.intents = []

        self.training_data = []

    def _prepare_intents_data(self, ignore_letters: tuple = ("!", "?", ",", ".")):
        documents = []

        for intent in self.intents_data["intents"]:
            if intent["tag"] not in self.intents:
                self.intents.append(intent["tag"])

            for pattern in intent["patterns"]:
                pattern_words = nltk.word_tokenize(pattern)
                self.words += pattern_words
                documents.append((pattern_words, intent["tag"]))
        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in ignore_letters]
        self.words = sorted(set(self.words))
        empty_output = [0] * len(self.intents)

        for document in documents:
            bag_of_words = []
            pattern_words = document[0]
            pattern_words = [self.lemmatizer.lemmatize(w.lower()) for w in pattern_words]
            for word in self.words:
                bag_of_words.append(1 if word in pattern_words else 0)

            output_row = empty_output.copy()
            output_row[self.intents.index(document[1])] = 1
            self.training_data.append([bag_of_words, output_row])

        random.shuffle(self.training_data)
        self.training_data = np.array(self.training_data, dtype="object")

        X = np.array([data[0] for data in self.training_data])
        y = np.array([data[1] for data in self.training_data])

        return X, y

    def fit_model(self, optimizer: Optimizer = None, epochs: int = 200):
        X, y = self._prepare_intents_data()

        if self.hidden_layers is None:
            self.model = Sequential()
            self.model.add(InputLayer(input_shape=(None, X.shape[1])))
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(y.shape[1], activation='softmax'))
        else:
            self.model = Sequential()
            self.model.add(InputLayer(input_shape=(None, X.shape[1])))
            for layer in self.hidden_layers:
                self.model.add(layer)
            self.model.add(Dense(y.shape[1], activation='softmax'))


        if optimizer is None:
            optimizer = Adam(learning_rate=0.01)

        self.model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        self.history = self.model.fit(X, y, epochs=epochs, batch_size=5, verbose=1)

    def save_model(self):
        self.model.save(f"{self.model_name}.keras", self.history)
        pickle.dump(self.words, open(f'{self.model_name}_words.pkl', 'wb'))
        pickle.dump(self.intents, open(f'{self.model_name}_intents.pkl', 'wb'))
    
    def load_model(self):
        self.model = load_model(f'{self.model_name}.keras')
        self.words = pickle.load(open(f'{self.model_name}_words.pkl', 'rb'))
        self.intents = pickle.load(open(f'{self.model_name}_intents.pkl', 'rb'))

    def _predict_intent(self, input_text: str):
        input_words = nltk.word_tokenize(input_text)
        input_words = [self.lemmatizer.lemmatize(w.lower()) for w in input_words]

        input_bag_of_words = [0] * len(self.words)
        
        for input_word in input_words:
            for i, word in enumerate(self.words):
                if input_word == word:
                    input_bag_of_words[i] = 1

        input_bag_of_words = np.array([input_bag_of_words])
        if np.all(input_bag_of_words == 0):
            predicted_intent = None
            return predicted_intent
        print(input_bag_of_words)
        predictions = self.model.predict(input_bag_of_words, verbose=0)[0]
        predicted_intent = self.intents[np.argmax(predictions)]
        print(predicted_intent)
        if predicted_intent is None:
            return("I don't understand. Please try again.")
        if predicted_intent == "app":
            import os
            try:
                if input_words[1] == "chrome":
                    os.system("start chrome")
                elif input_words[1] == "edge":
                    os.system("start msedge")
            except IndexError:
                print("Give Something to open")
                return 
            
        self.max_prob = np.max(predictions)
        print(self.max_prob)
        return predicted_intent

    def process_input(self, input_text: str):
        predicted_intent = self._predict_intent(input_text)

        try:
            if not predicted_intent:
                return "I'm sorry, I don't have information on that. Can you try something else?"
            elif predicted_intent in self.method_mappings:
                self.method_mappings[predicted_intent]()

            for intent in self.intents_data["intents"]:
                if intent["tag"] == predicted_intent:
                    return random.choice(intent["responses"])
        except IndexError:
            return "I don't understand. Please try again."



if __name__ == "__main__":
    assistant = Assistant('intents.json')
    assistant.fit_model(epochs=50)
    assistant.save_model()

    done = False
    while not done:
        message = input("Enter a message: ")
        if message == "STOP" or message == "stop":
            done = True
        else:
            print(assistant.process_input(message))

#using Torch
'''
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
# nltk.download('punkt')
# nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

with open("intents.json") as f:
    intents_data = json.load(f)

bag_of_words = []
tags = []
xy = []

for intent in intents_data["intents"]:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        nltk_tokens = nltk.word_tokenize(pattern)
        bag_of_words.extend(nltk_tokens)
        xy.append((nltk_tokens, tag))

ignore_words = ['!', '?', ',', '.']
bag_of_words = [lemmatizer.lemmatize(w.lower()) for w in bag_of_words if w not in ignore_words]
bag_of_words = sorted(set(bag_of_words))
tags = sorted(set(tags))

def check(tokens, words):
    bag = np.zeros(len(words), dtype=np.int64)
    for token in tokens:
        if token in words:
            bag[words.index(token)] = 1
    return bag

x = []
y = []
sen = ['hi','hello','world']
c = ['hi','hello','brave']
z = check(sen,c)
print(z)
for (pattern, tag) in xy:
    bag = check(pattern, bag_of_words)
    x.append(bag)
    y.append(tags.index(tag))

x = np.array(x)
y = np.array(y)

print("X:", x)
print("Y:", y)
print("Tags:", tags)
class c_Dataset(Dataset):
    def __init__(self):
        self.n = len(x)
        self.x_data=x
        self.y_data=y
    def __getitem__(self, index):
        return self.x_data[index],self.y_data
data=c_Dataset()
train_loader = DataLoader(dataset = data,batch_size=8,shuffle=True,num_workers=2)
'''