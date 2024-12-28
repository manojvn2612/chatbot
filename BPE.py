import re
from collections import Counter,defaultdict
import numpy as np

#basic bpe algo which will be used for splitting at the space and emoving the use of <\w> token in the training process.
class BPE:

    #initial info like making vocab,words for training and traning epochs etc
    def __init__(self,epoch = 50):
        self.vocab = defaultdict(int)
        self.epochs = epoch
        self.curr = 2
        self.vocab.update({
            "<pad>" : 0, 
            "<unk>" : 1
        })
        self.words = []

    #tokenize the words into characters to intitalize trainings
    def tokenize(self,sent):
        words = re.findall("(\S+|\s)", sent)
        token = []
        token.append(list(words[0]))
        i = 1
        while i < len(words)-1:
                if words[i] == " " and words[i+1] == " ":#used for only space without any successing words
                    j = i
                    while j < len(words) and words[j+1] == " ":
                        j += 1
                    token.append(list([" ".join(words[i:j])]))
                    i = j
                if words[i] == " " and words[i+1] != " ":
                    token.append(list(" " + words[i+1]))
                i += 1

        self.words = np.array(token,dtype=object)
        for word in token:
            for char in word:
                if char not in self.vocab:
                    self.vocab[char] = self.curr
                    self.curr += 1
        return token
    
    #get the pairs i.e adjacent characters
    def get_pair(self):
        pair = Counter()
        for words in self.words:
            for i in range(len(words)-1):
                pair[(words[i],words[i+1])] = pair.get((words[i],words[i+1]),0) + 1
        return pair

    #merge the pairs to form a new word and update the vocabulary and words in the text
    def merge_pair(self, pair):
        new_words = []
        for word in self.words:

            merged = []
            i = 0
            while i < len(word) - 1:
                if (word[i], word[i + 1]) == pair:
                    merged.append("".join(pair))
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            if i < len(word):
                merged.append(word[-1])
            new_words.append(merged)
        self.words = np.array(new_words, dtype=object)

    #training the bpe model on the text input. This will update the vocabulary and words in the text. 1000 epochs are used by default. 1000 epochs are usually enough for training a good BPE model. 1000 epochs can be adjusted based on the resources available. 1000 epochs may not be enough for very large corpora. 1000 epochs are used here for simplicity and demonstration purposes.
    def train(self, text):
        self.tokenize(text)
        for _ in range(self.epochs):
            pair_counts = self.get_pair()
            if not pair_counts:
                break
            most_common_pair = pair_counts.most_common(1)[0][0]
            self.merge_pair(most_common_pair)
            # Add new token to the vocabulary
            new_token = "".join(most_common_pair)
            if new_token not in self.vocab:
                self.vocab[new_token] = self.curr
                self.curr += 1
        for word in self.words:
            for i, token in enumerate(word):
                if token not in self.vocab:
                    word[i] = "<unk>"

    #encode the vocabulary
    def encode(self, text):
        self.tokenize(text)
        for _ in range(self.epochs):
            pair_counts = self.get_pair()
            if not pair_counts:
                break
            most_common_pair = pair_counts.most_common(1)[0][0]
            self.merge_pair(most_common_pair)
        return [" ".join(word) for word in self.words]

#test run
if __name__ == "__main__":
    bpe = BPE(epoch=1000)
    text = "This is a sample text. This text is a    sample. sa"
    bpe.train(text)
    print("Vocabulary:", bpe.vocab)
    print("Encoded words:", bpe.encode("T"))