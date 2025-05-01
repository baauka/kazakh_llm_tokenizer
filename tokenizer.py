import re

class KazakhLMTokenizer:
    
    def __init__(self, corpus_txt, desired_vocab_size):
        self.text = self.load_corpus(corpus_txt)
        self.desired_vocab_size = desired_vocab_size
        self.merges = {}

        # Filter tokens and initialize tokenizer parameters
        self.tokens = self.filter_tokens(self.text)
        # self.vocab_size = len(self.tokens)
        self.last_token_idx = max(self.tokens)

        # Calculate number of merges required to reach desired vocab size
        self.num_merges = self.desired_vocab_size - 256

        # Initialize vocabulary with 256 tokens
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

    def load_corpus(self, corpus_txt):
        """Load and read the text corpus."""
        
        with open(corpus_txt, 'r', encoding='utf-8') as file:
            return file.read()

    def filter_tokens(self, text):
        """Filter tokens based on regex to exclude unwanted characters."""
        
        pattern = r'^(?!.*[?!\d]).*$'
        filtered_tokens = [token for token in text if re.match(pattern, token)]
        return "".join(filtered_tokens).encode('utf-8')

    def get_pair_counts(self, ids):
        """Count token pairs."""
        
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge_token_pair(self, ids, pair, idx):
        """Merge the most frequent token pair into a new token."""
        
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:  
                newids.append(idx)
                i += 2  
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def train_tokenizer(self):
        """Train the tokenizer by merging frequent token pairs."""
        
        print("Training process has started...")

        for i in range(self.num_merges):
            stats = self.get_pair_counts(self.tokens)
            top_pair = max(stats, key=stats.get)
            new_idx = 256 + i
            print(f"Merging {top_pair} into a new token {new_idx}. Token number = {i+1}")
            self.merges[top_pair] = new_idx
            self.tokens = self.merge_token_pair(self.tokens, top_pair, new_idx)

            self.vocab[new_idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            
    def encode(self, text):
        """Encodes the input text into a sequence of tokens by applying Byte Pair Encoding (BPE)."""
        
        tokens = list(text.encode('utf-8'))
        while len(tokens) >= 2:
            stats = self.get_pair_counts(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break 
            idx = self.merges[pair]
            tokens = self.merge_token_pair(tokens, pair, idx)
        return tokens
            
            
    def decode(self, ids):
        """Decodes a sequence of token IDs back into the original text."""
        
        tokens = "".join(self.vocab[idx] for idx in ids)
        text = tokens.decode('utf-8', errors='replace')
        return text 
    
    