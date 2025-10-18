import re
from typing import List, Dict, Tuple, Any
from collections import defaultdict, Counter
import json

class BPETokenizer:
    """Byte Pair Encoding tokenizer for romance text"""

    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.vocab: Dict[str, int] = {}
        self.inverse_vocab: Dict[int, str] = {}
        self.merges: Dict[Tuple[str, str], str] = {}

        # Special tokens
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"

        self.pad_token_id = 0
        self.unk_token_id = 1
        self.bos_token_id = 2
        self.eos_token_id = 3

        # Initialize with base vocabulary (characters)
        self._init_base_vocab()

    def _init_base_vocab(self):
        """Initialize base vocabulary with printable ASCII characters"""
        self.vocab = {}
        self.inverse_vocab = {}

        # Add special tokens
        self.vocab[self.pad_token] = self.pad_token_id
        self.vocab[self.unk_token] = self.unk_token_id
        self.vocab[self.bos_token] = self.bos_token_id
        self.vocab[self.eos_token] = self.eos_token_id

        # Add printable ASCII characters
        idx = 4
        for i in range(32, 127):  # Printable ASCII
            char = chr(i)
            self.vocab[char] = idx
            idx += 1

        # Update inverse vocab
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def train(self, text: str, num_merges: int = 40000):
        """Train BPE on the given text"""
        # Preprocessing: split into words and add spaces
        words = re.findall(r'\S+', text.lower())
        word_freq = Counter(words)

        # Initialize vocabulary with characters
        vocab = set()
        for word in words:
            for char in word:
                vocab.add(char)

        # Convert words to character sequences
        word_tokens = {}
        for word, freq in word_freq.items():
            tokens = list(word)
            word_tokens[word] = (tokens, freq)

        # BPE training
        merges = {}
        for i in range(num_merges):
            # Count pair frequencies
            pair_counts = defaultdict(int)
            for word, (tokens, freq) in word_tokens.items():
                for j in range(len(tokens) - 1):
                    pair = (tokens[j], tokens[j + 1])
                    pair_counts[pair] += freq

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            new_token = ''.join(best_pair)

            # Merge the pair
            merges[best_pair] = new_token

            # Update word tokens
            new_word_tokens = {}
            for word, (tokens, freq) in word_tokens.items():
                new_tokens = []
                j = 0
                while j < len(tokens):
                    if j < len(tokens) - 1 and (tokens[j], tokens[j + 1]) == best_pair:
                        new_tokens.append(new_token)
                        j += 2
                    else:
                        new_tokens.append(tokens[j])
                        j += 1
                new_word_tokens[word] = (new_tokens, freq)
            word_tokens = new_word_tokens

        self.merges = merges

        # Build final vocabulary
        self._build_vocab_from_merges()

    def _build_vocab_from_merges(self):
        """Build vocabulary from merges"""
        idx = len(self.vocab)
        for merge in self.merges.values():
            if merge not in self.vocab:
                self.vocab[merge] = idx
                idx += 1

        # Limit to vocab_size
        if len(self.vocab) > self.vocab_size:
            # Keep most frequent (simplified)
            sorted_vocab = sorted(self.vocab.items(), key=lambda x: len(x[0]), reverse=True)
            self.vocab = dict(sorted_vocab[:self.vocab_size])

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        if not self.vocab:
            # Fallback to character-level
            return [self.vocab.get(char, self.unk_token_id) for char in text]

        # Preprocessing
        text = text.lower()
        words = re.findall(r'\S+', text)

        tokens = []
        for word in words:
            # Apply BPE merges
            word_tokens = list(word)
            while len(word_tokens) > 1:
                # Find best merge
                pairs = [(word_tokens[j], word_tokens[j + 1]) for j in range(len(word_tokens) - 1)]
                merge_candidates = [pair for pair in pairs if pair in self.merges]

                if not merge_candidates:
                    break

                # Apply first available merge (simplified)
                pair = merge_candidates[0]
                new_token = self.merges[pair]

                # Merge in word_tokens
                new_word_tokens = []
                i = 0
                while i < len(word_tokens):
                    if i < len(word_tokens) - 1 and (word_tokens[i], word_tokens[i + 1]) == pair:
                        new_word_tokens.append(new_token)
                        i += 2
                    else:
                        new_word_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_word_tokens

            # Convert to IDs
            for token in word_tokens:
                token_id = self.vocab.get(token, self.unk_token_id)
                tokens.append(token_id)

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                tokens.append(self.inverse_vocab[token_id])
            else:
                tokens.append(self.unk_token)

        # Join tokens
        text = ''.join(tokens)

        # Post-processing
        text = re.sub(r'([.!?])', r' \1', text)  # Add spaces before punctuation
        text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces

        return text

    def save_vocab(self, path: str):
        """Save vocabulary to file"""
        with open(path, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'merges': {f"{k[0]} {k[1]}": v for k, v in self.merges.items()}
            }, f, indent=2)

    def load_vocab(self, path: str):
        """Load vocabulary from file"""
        with open(path, 'r') as f:
            data = json.load(f)
            self.vocab = data['vocab']
            self.merges = {tuple(k.split()): v for k, v in data['merges'].items()}
            self.inverse_vocab = {v: k for k, v in self.vocab.items()}


class RomanceDatasetLoader:
    """Loader for romance novel datasets"""

    def __init__(self):
        self.datasets = []

    def load_from_text(self, text: str) -> str:
        """Load and preprocess text data"""
        # Basic preprocessing for romance text
        text = text.lower()

        # Filter for fashion/beauty content (optional)
        fashion_keywords = ['dress', 'fashion', 'makeup', 'beauty', 'perfume', 'lipstick', 'style']
        lines = text.split('\n')
        filtered_lines = []

        for line in lines:
            if any(keyword in line for keyword in fashion_keywords):
                filtered_lines.append(line)

        if filtered_lines:
            return '\n'.join(filtered_lines)
        else:
            return text  # Return original if no fashion content found

    def add_dataset(self, name: str, text: str):
        """Add a dataset"""
        self.datasets.append({'name': name, 'text': text})


class TokenIO:
    """Token input/output utilities"""

    def __init__(self, tokenizer: BPETokenizer):
        self.tokenizer = tokenizer

    def save_tokens(self, tokens: List[int], path: str):
        """Save tokens to file"""
        with open(path, 'w') as f:
            json.dump(tokens, f)

    def load_tokens(self, path: str) -> List[int]:
        """Load tokens from file"""
        with open(path, 'r') as f:
            return json.load(f)

    def tokens_to_text(self, tokens: List[int]) -> str:
        """Convert tokens to readable text"""
        return self.tokenizer.decode(tokens)

    def text_to_tokens(self, text: str) -> List[int]:
        """Convert text to tokens"""
        return self.tokenizer.encode(text)