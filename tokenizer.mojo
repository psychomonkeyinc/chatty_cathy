from collections import List
from pathlib import Path
import json

# Fast BPE tokenizer for Mojo (based on HuggingFace implementation concept)
struct BPETokenizer:
    """Fast Byte-Pair Encoding tokenizer optimized for romance novels"""
    var vocab: Dict[String, Int]
    var merges: List[Tuple[String, String]]
    var vocab_size: Int
    var unk_token: String
    var special_tokens: Dict[String, Int]
    
    fn __init__(inout self, vocab_size: Int = 50000):
        self.vocab_size = vocab_size
        self.vocab = Dict[String, Int]()
        self.merges = List[Tuple[String, String]]()
        self.unk_token = "<UNK>"
        self.special_tokens = Dict[String, Int]()
        
        # Fashion/beauty/emotion special tokens for romance novels
        var specials = [
            "<UNK>", "<PAD>", "<BOS>", "<EOS>",
            "<FASHION>", "<MAKEUP>", "<BEAUTY>", "<DESIRE>",
            "<LOVE>", "<LUST>", "<FEAR>", "<JOY>",
            "<SUBCONSCIOUS>", "<UNCONSCIOUS>", "<MEMORY>"
        ]
        
        for i in range(len(specials)):
            self.special_tokens[specials[i]] = i
            self.vocab[specials[i]] = i
    
    fn train_from_text(inout self, text: String):
        """Train BPE on romance novel text - fast greedy algorithm"""
        # Step 1: Initialize with byte-level vocabulary (256 chars)
        for i in range(256):
            var byte_char = chr(i)
            self.vocab[byte_char] = len(self.vocab)
        
        # Step 2: Split text into words and get frequencies
        var word_freqs = self._get_word_frequencies(text)
        
        # Step 3: Iteratively merge most frequent pairs
        var num_merges = self.vocab_size - len(self.vocab)
        for merge_iter in range(num_merges):
            var pair_freqs = self._count_pairs(word_freqs)
            if len(pair_freqs) == 0:
                break
            
            var best_pair = self._get_most_frequent_pair(pair_freqs)
            var merged = best_pair[0] + best_pair[1]
            
            self.merges.append(best_pair)
            self.vocab[merged] = len(self.vocab)
            
            # Update word frequencies with merged pair
            word_freqs = self._merge_pair(word_freqs, best_pair, merged)
    
    fn _get_word_frequencies(self, text: String) -> Dict[String, Int]:
        """Split text into words and count frequencies"""
        var freqs = Dict[String, Int]()
        var words = text.split()
        
        for word in words:
            var word_with_end = word + "</w>"
            if word_with_end in freqs:
                freqs[word_with_end] += 1
            else:
                freqs[word_with_end] = 1
        
        return freqs
    
    fn encode(self, text: String) -> List[Int]:
        """Encode text to token IDs using trained BPE"""
        var tokens = List[Int]()
        var words = text.split()
        
        for word in words:
            var word_tokens = self._encode_word(word + "</w>")
            for token_id in word_tokens:
                tokens.append(token_id)
        
        return tokens
    
    fn _encode_word(self, word: String) -> List[Int]:
        """Encode a single word using BPE merges"""
        # Start with character-level
        var splits = List[String]()
        for i in range(len(word)):
            splits.append(String(word[i]))
        
        # Apply merge rules in order
        for merge in self.merges:
            var i = 0
            while i < len(splits) - 1:
                if splits[i] == merge[0] and splits[i+1] == merge[1]:
                    var merged = merge[0] + merge[1]
                    splits[i] = merged
                    splits.pop(i+1)
                else:
                    i += 1
        
        # Convert to IDs
        var ids = List[Int]()
        for token in splits:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab[self.unk_token])
        
        return ids
    
    fn decode(self, token_ids: List[Int]) -> String:
        """Decode token IDs back to text"""
        var text = String("")
        var id_to_token = self._create_reverse_vocab()
        
        for token_id in token_ids:
            if token_id in id_to_token:
                var token = id_to_token[token_id]
                text += token.replace("</w>", " ")
        
        return text.strip()
    
    fn save_vocab(self, filepath: String):
        """Save vocabulary to JSON file"""
        # Placeholder - implement with Mojo file I/O
        print("Saving vocab to:", filepath)
    
    fn load_vocab(inout self, filepath: String):
        """Load vocabulary from JSON file"""
        # Placeholder - implement with Mojo file I/O
        print("Loading vocab from:", filepath)


struct RomanceDatasetLoader:
    """Load and preprocess romance novel datasets with fashion/beauty focus"""
    var tokenizer: BPETokenizer
    var cache_dir: String
    
    fn __init__(inout self, tokenizer: BPETokenizer):
        self.tokenizer = tokenizer
        self.cache_dir = "./data/romance_novels"
    
    fn download_goodreads_romance(self) -> String:
        """Download Goodreads romance dataset
        Source: https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html
        335,449 romance books with 3.5M reviews
        """
        print("Download from: https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html")
        print("Use goodreads_reviews_romance.json.gz")
        return self.cache_dir + "/goodreads_romance.txt"
    
    fn download_gutenberg_romance(self) -> String:
        """Download Project Gutenberg romance novels
        Focus on: Fashion, beauty, risque content
        Source: https://www.gutenberg.org/ebooks/subject/2487
        """
        print("Download from: https://www.kaggle.com/datasets/mateibejan/15000-gutenberg-books")
        print("Filter for romance genre")
        return self.cache_dir + "/gutenberg_romance.txt"
    
    fn filter_fashion_beauty_content(self, text: String) -> String:
        """Extract passages focused on fashion, makeup, beauty"""
        # Look for keywords: dress, gown, lipstick, perfume, hair, makeup, beauty
        # This would use regex or string matching in real implementation
        var keywords = ["dress", "gown", "fashion", "lipstick", "makeup", 
                       "beauty", "perfume", "hair", "silk", "lace", "elegant"]
        
        # Placeholder: In real impl, extract relevant paragraphs
        return text
    
    fn load_dataset(self, filepath: String) -> String:
        """Load text from file"""
        # Placeholder for file I/O
        var sample_text = """
        She slipped into the crimson silk dress, the fabric caressing her skin 
        like a lover's touch. The makeup artist had done wonders - her lips were 
        painted a deep burgundy, matching the dress perfectly. She felt powerful, 
        beautiful, dangerous. His eyes would be on her tonight, and she knew it.
        
        The perfume she wore was intoxicating, a mix of jasmine and vanilla that 
        left a trail wherever she walked. Her hair cascaded down her back in loose 
        curls, each strand catching the light. She was ready for whatever the night 
        would bring.
        """ * 100  # Repeat for training data
        
        return sample_text


struct TokenIO:
    """Main I/O interface for tokenization"""
    var tokenizer: BPETokenizer
    var dataset_loader: RomanceDatasetLoader

    fn __init__(inout self, vocab_size: Int = 50000):
        self.tokenizer = BPETokenizer(vocab_size)
        self.dataset_loader = RomanceDatasetLoader(self.tokenizer)

    fn train_on_romance_corpus(inout self):
        """Train tokenizer on romance novel corpus"""
        print("\n=== Training BPE Tokenizer on Romance Novels ===")
        
        # Load datasets
        var text = self.dataset_loader.load_dataset("romance_corpus.txt")
        
        # Focus on fashion/beauty content
        text = self.dataset_loader.filter_fashion_beauty_content(text)
        
        print("Training on", len(text), "characters")
        self.tokenizer.train_from_text(text)
        print("Vocabulary size:", len(self.tokenizer.vocab))
        print("Training complete!")
    
    fn tokenize_input(self, text: String) -> List[Int]:
        return self.tokenizer.encode(text)

    fn detokenize_output(self, tokens: List[Int]) -> String:
        return self.tokenizer.decode(tokens)