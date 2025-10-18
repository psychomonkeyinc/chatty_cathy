import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any, Optional
import math
import random
import json

# Python equivalent of cognition_core.mojo
# Using PyTorch for tensor operations

class EmotionalState:
    """Track conscious and subconscious emotional states"""
    def __init__(self):
        self.conscious_emotions: Dict[str, float] = {}
        self.subconscious_emotions: Dict[str, float] = {}
        self.unconscious_patterns: List[str] = []
        self.emotional_memory: List[Tuple[str, float]] = []

        # Initialize base emotions
        emotions = ["desire", "love", "lust", "fear", "joy", "anxiety", "confidence"]
        for emotion in emotions:
            self.conscious_emotions[emotion] = 0.0
            self.subconscious_emotions[emotion] = 0.0

    def update_from_text(self, text: str):
        """Analyze text and update emotional state"""
        # Track fashion/beauty mentions
        if "dress" in text.lower() or "fashion" in text.lower():
            self.conscious_emotions["confidence"] += 0.2

        if "makeup" in text.lower() or "lipstick" in text.lower():
            self.subconscious_emotions["desire"] += 0.15

        if "touch" in text.lower() or "caress" in text.lower():
            self.subconscious_emotions["lust"] += 0.3

        # Store in memory
        self.emotional_memory.append((text[:50], 0.8))

    def predict_next_emotion(self) -> str:
        """Predict likely next emotional state based on patterns"""
        # Find dominant subconscious emotion
        max_emotion = "neutral"
        max_value = 0.0

        for emotion, value in self.subconscious_emotions.items():
            if value > max_value:
                max_value = value
                max_emotion = emotion

        return max_emotion


class chatty_cathyMemoryCore:
    """Memory system for the chatty_cathy model"""
    def __init__(self):
        self.short_term: List[str] = []
        self.fashion_memory: List[str] = []
        self.beauty_preferences: Dict[str, float] = {}
        self.max_short_term = 100

    def add_to_short_term(self, item: str):
        """Add item to short-term memory"""
        self.short_term.append(item)
        if len(self.short_term) > self.max_short_term:
            self.short_term.pop(0)

    def add_fashion_memory(self, item: str):
        """Add fashion-related memory"""
        self.fashion_memory.append(item)

    def get_fashion_context(self) -> str:
        """Get fashion context for generation"""
        if self.fashion_memory:
            return " ".join(self.fashion_memory[-5:])  # Last 5 fashion memories
        return ""


class chatty_cathyCognitionCore:
    """chatty_cathy v1 Cognition Core - Female personality with emotional processing"""
    def __init__(self):
        self.memory = chatty_cathyMemoryCore()
        self.emotional_state = EmotionalState()
        self.personality_traits = {
            "feminine": 0.9,
            "sensual": 0.8,
            "confident": 0.75,
            "empathetic": 0.7,
            "creative": 0.8
        }

    def process_input(self, text: str) -> Dict[str, Any]:
        """Process input text through cognition core"""
        # Update emotional state
        self.emotional_state.update_from_text(text)

        # Add to memory
        self.memory.add_to_short_term(text)

        # Check for fashion/beauty content
        fashion_keywords = ["dress", "fashion", "makeup", "beauty", "style", "perfume"]
        if any(keyword in text.lower() for keyword in fashion_keywords):
            self.memory.add_fashion_memory(text)

        return {
            "emotional_state": self.emotional_state.predict_next_emotion(),
            "personality_influence": self._get_personality_influence(),
            "memory_context": self.memory.get_fashion_context()
        }

    def _get_personality_influence(self) -> Dict[str, float]:
        """Get personality influence for generation"""
        return self.personality_traits.copy()


class TransformerBlock(nn.Module):
    """Simple transformer block for the language model"""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x


class chatty_cathyLanguageModel(nn.Module):
    """Language model with chatty_cathy v1 cognition core integration"""
    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int = 4, num_heads: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(1000, embed_dim)  # Max sequence length

        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, embed_dim * 4)
            for _ in range(num_layers)
        ])

        self.head = nn.Linear(embed_dim, vocab_size)
        self.cognition = chatty_cathyCognitionCore()

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: List[int]) -> torch.Tensor:
        """Forward pass through the model"""
        # Convert to tensor
        x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)  # Add batch dimension

        # Embeddings
        seq_len = x.size(1)
        pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(pos_ids)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output logits
        logits = self.head(x)

        return logits

    def generate(self, prompt: str, tokenizer, max_length: int = 50, temperature: float = 0.8) -> str:
        """Generate text with cognition influence"""
        # Tokenize prompt
        tokens = tokenizer.encode(prompt)

        for _ in range(max_length):
            # Get logits
            logits = self.forward(tokens)

            # Get last token logits
            next_logits = logits[0, -1, :] / temperature

            # Apply softmax
            probs = F.softmax(next_logits, dim=-1)

            # Sample next token
            next_token = torch.multinomial(probs, 1).item()

            # Add to sequence
            tokens.append(next_token)

            # Check for EOS or stop conditions
            if next_token == tokenizer.eos_token_id:
                break

        # Decode
        generated_text = tokenizer.decode(tokens)
        return generated_text

    def save(self, path: str):
        """Save model weights"""
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, vocab_size: int, embed_dim: int):
        """Load model weights"""
        model = cls(vocab_size, embed_dim)
        model.load_state_dict(torch.load(path))
        return model