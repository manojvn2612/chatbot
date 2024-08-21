from collections import defaultdict, Counter

class BPE:
    def __init__(self, text):
        
        self.words = list(text)  
        self.vocab = set(self.words)  
    
    def get_pair(self):
        # Count frequency of adjacent pairs
        pair_freq = defaultdict(int)
        for i in range(len(self.words) - 1):
            pair = (self.words[i], self.words[i + 1])
            pair_freq[pair] += 1
        # Return the most frequent pair
        return max(pair_freq, key=pair_freq.get) if pair_freq else None
    
    def merge_pair(self, pair):
        # Merge the most frequent pair in the entire sequence
        new_words = []
        i = 0
        while i < len(self.words) - 1:
            if (self.words[i], self.words[i + 1]) == pair:
                new_words.append("".join(pair))  # Merge the pair into one token
                i += 2  # Skip the next token as it has been merged
            else:
                new_words.append(self.words[i])
                i += 1
        if i < len(self.words):
            new_words.append(self.words[-1])
        self.words = new_words
        self.vocab.add("".join(pair))  # Add the merged token to the vocabulary
    
    def fit(self, iterations):
        # Perform the merge operation for the specified number of iterations
        for _ in range(iterations):
            pair = self.get_pair()
            if not pair:
                break  
            self.merge_pair(pair)
        print("Final words:", self.words)
        print("Vocabulary:", sorted(self.vocab))

if __name__ == "__main__":
    text = '''def add_numbers(a, b):\n\treturn a + b'''
    bpe = BPE(text)
    bpe.fit(100)
