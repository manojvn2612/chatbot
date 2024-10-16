import os
import json
import random
import pickle

from typing import Union
import nltk
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, InputLayer, Flatten
from tensorflow.keras.optimizers import Adam, Optimizer

class Assistant:

    def __init__(self, intents_data: Union[str, os.PathLike, dict], method_mappings: dict = {}, hidden_layers: list = None, model_name: str = "basic_model") -> None:
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
        print(documents)
        self.words = [self.lemmatizer.lemmatize(w.lower()) for w in self.words if w not in ignore_letters]
        
        print(self.words)
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
        print(self.training_data)
        X = np.array([data[0] for data in self.training_data])
        y = np.array([data[1] for data in self.training_data])
        print(len(X))
        return X, y

    def fit_model(self, optimizer: Optimizer = None, epochs: int = 200):
        X, y = self._prepare_intents_data()

        # Reshape X to (num_samples, sequence_length, num_features)
        X = X.reshape(X.shape[0], 1, X.shape[1])  # Add a sequence dimension

        if self.hidden_layers is None:
            self.model = Sequential()
            self.model.add(InputLayer(input_shape=(1, X.shape[2])))  # Update input shape
            self.model.add(Dense(128, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Flatten())  # Add Flatten layer to remove extra dimension
            self.model.add(Dense(y.shape[1], activation='softmax'))  # Output layer
        else:
            self.model = Sequential()
            self.model.add(InputLayer(input_shape=(1, X.shape[2])))  # Update input shape
            for layer in self.hidden_layers:
                self.model.add(layer)
            self.model.add(Flatten())  # Add Flatten layer to remove extra dimension
            self.model.add(Dense(y.shape[1], activation='softmax'))  # Output layer

        if optimizer is None:
            optimizer = Adam(learning_rate=0.01)

        self.model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

        self.history = self.model.fit(X, y, epochs=epochs, batch_size=5, verbose=1)

    def save_model(self):
        self.model.save(f"{self.model_name}.h5", self.history)
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

        # Reshape to match expected input shape (1, 1, 43)
        input_bag_of_words = input_bag_of_words.reshape(1, 1, -1)

        predictions = self.model.predict(input_bag_of_words, verbose=0)[0]
        print(predictions)
        predicted_intent = self.intents[np.argmax(predictions)]
        max_prob = np.max(predictions)
        return predicted_intent


    def process_input(self, input_text: str):
        predicted_intent = self._predict_intent(input_text)

        try:
            if predicted_intent in self.method_mappings:
                self.method_mappings[predicted_intent]()

            for intent in self.intents_data["intents"]:
                if intent["tag"] == predicted_intent:
                    return random.choice(intent["responses"])
        except IndexError:
            return "I don't understand. Please try again."


if __name__ == "__main__":
    assistant = Assistant('F:\ChatBotUI\python\intents.json')
    assistant.fit_model(epochs=50)
    assistant.save_model()

    done = False
    while not done:
        message = input("Enter a message: ")
        if message == "STOP":
            done = True
        else:
            print(assistant.process_input(message))
