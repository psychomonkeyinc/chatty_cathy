from tensor import Tensor, TensorShape
from collections import List
from math import sqrt, exp
from random import random_float64
from input_token_output import TokenIO, BPETokenizer

# Cognition Core Components
struct EmotionalState:
    """Track conscious and subconscious emotional states"""
    var conscious_emotions: Dict[String, Float32]  # Surface emotions
    var subconscious_emotions: Dict[String, Float32]  # Hidden desires/fears
    var unconscious_patterns: List[String]  # Deep behavioral patterns
    var emotional_memory: List[Tuple[String, Float32]]  # Context + intensity
    
    fn __init__(inout self):
        self.conscious_emotions = Dict[String, Float32]()
        self.subconscious_emotions = Dict[String, Float32]()
        self.unconscious_patterns = List[String]()
        self.emotional_memory = List[Tuple[String, Float32]]()
        
        # Initialize base emotions
        var emotions = ["desire", "love", "lust", "fear", "joy", "anxiety", "confidence"]
        for emotion in emotions:
            self.conscious_emotions[emotion] = 0.0
            self.subconscious_emotions[emotion] = 0.0
    
    fn update_from_text(inout self, text: String):
        """Analyze text and update emotional state"""
        # Track fashion/beauty mentions
        if "dress" in text or "fashion" in text:
            self.conscious_emotions["confidence"] += 0.2
        
        if "makeup" in text or "lipstick" in text:
            self.subconscious_emotions["desire"] += 0.15
        
        if "touch" in text or "caress" in text:
            self.subconscious_emotions["lust"] += 0.3
        
        # Store in memory
        self.emotional_memory.append((text[:50], 0.8))
    
    fn predict_next_emotion(self) -> String:
        """Predict likely next emotional state based on patterns"""
        # Find dominant subconscious emotion
        var max_emotion = "neutral"
        var max_value = Float32(0.0)
        
        for emotion, value in self.subconscious_emotions.items():
            if value > max_value:
                max_value = value
                max_emotion = emotion
        
        return max_emotion


struct chatty_cathyMemoryCore:
    """Memory system with emotional tagging"""
    var short_term: List[Tuple[String, EmotionalState]]
    var long_term: Dict[String, List[String]]  # Topic -> memories
    var fashion_memory: List[String]  # Specific fashion/beauty memories
    var beauty_preferences: Dict[String, Float32]
    
    fn __init__(inout self):
        self.short_term = List[Tuple[String, EmotionalState]]()
        self.long_term = Dict[String, List[String]]()
        self.fashion_memory = List[String]()
        self.beauty_preferences = Dict[String, Float32]()
        
        # Initialize preferences
        self.beauty_preferences["elegant"] = 0.8
        self.beauty_preferences["bold"] = 0.6
        self.beauty_preferences["subtle"] = 0.3
    
    fn store_memory(inout self, text: String, emotion: EmotionalState):
        """Store memory with emotional context"""
        self.short_term.append((text, emotion))
        
        # Extract fashion/beauty content to special memory
        if "dress" in text or "makeup" in text or "beauty" in text:
            self.fashion_memory.append(text)
    
    fn recall_relevant(self, query: String) -> List[String]:
        """Recall memories relevant to query"""
        var relevant = List[String]()
        
        # Check fashion memory first
        for memory in self.fashion_memory:
            if any(word in memory for word in query.split()):
                relevant.append(memory)
        
        return relevant


struct chatty_cathyCognitionCore:
    """
    chatty_cathy v1 Cognition Core - Female personality with:
    - Conscious/subconscious/unconscious emotional layers
    - Fashion, makeup, beauty focus
    - Emotional memory and prediction
    """
    var emotional_state: EmotionalState
    var memory: chatty_cathyMemoryCore
    var personality_traits: Dict[String, Float32]
    
    fn __init__(inout self):
        self.emotional_state = EmotionalState()
        self.memory = chatty_cathyMemoryCore()
        self.personality_traits = Dict[String, Float32]()
        
        # Define chatty_cathy's personality
        self.personality_traits["feminine"] = 0.95
        self.personality_traits["fashion_conscious"] = 0.9
        self.personality_traits["emotionally_aware"] = 0.85
        self.personality_traits["sensual"] = 0.8
        self.personality_traits["confident"] = 0.75
        self.personality_traits["mysterious"] = 0.7
    
    fn process_input(inout self, text: String) -> Dict[String, Any]:
        """Process input through cognition layers"""
        # Update emotional state
        self.emotional_state.update_from_text(text)
        
        # Store in memory
        self.memory.store_memory(text, self.emotional_state)
        
        # Predict emotional trajectory
        var predicted_emotion = self.emotional_state.predict_next_emotion()
        
        # Recall relevant memories
        var relevant_memories = self.memory.recall_relevant(text)
        
        return {
            "conscious_state": self.emotional_state.conscious_emotions,
            "subconscious_state": self.emotional_state.subconscious_emotions,
            "predicted_emotion": predicted_emotion,
            "memories": relevant_memories,
            "personality_influence": self.personality_traits
        }
    
    fn generate_emotional_context(self, base_text: String) -> String:
        """Add emotional depth to generated text"""
        var context = base_text
        
        # Add subconscious layer
        var dominant_emotion = self.emotional_state.predict_next_emotion()
        
        if dominant_emotion == "desire":
            context += " Her heart raced with anticipation."
        elif dominant_emotion == "confidence":
            context += " She felt powerful and beautiful."
        elif dominant_emotion == "lust":
            context += " A shiver of pleasure ran through her."
        
        return context


# Neural network components (simplified for speed)
struct Linear:
    var weight: Tensor[DType.float32]
    var bias: Tensor[DType.float32]
    
    fn __init__(inout self, in_dim: Int, out_dim: Int):
        self.weight = Tensor[DType.float32](TensorShape(in_dim, out_dim))
        self.bias = Tensor[DType.float32](TensorShape(out_dim))
        
        var scale = Float32(sqrt(2.0 / Float64(in_dim)))
        for i in range(in_dim):
            for j in range(out_dim):
                self.weight[i, j] = Float32((random_float64() - 0.5) * 2.0) * scale
        for i in range(out_dim):
            self.bias[i] = 0.0


struct Embedding:
    var weight: Tensor[DType.float32]
    
    fn __init__(inout self, vocab_size: Int, embed_dim: Int):
        self.weight = Tensor[DType.float32](TensorShape(vocab_size, embed_dim))
        for i in range(vocab_size):
            for j in range(embed_dim):
                self.weight[i, j] = Float32((random_float64() - 0.5) * 0.04)


struct RomanceTransformer:
    """Transformer model optimized for romance novel generation"""
    var embed: Embedding
    var head: Linear
    var vocab_size: Int
    var embed_dim: Int
    
    fn __init__(inout self, vocab_size: Int, embed_dim: Int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed = Embedding(vocab_size, embed_dim)
        self.head = Linear(embed_dim, vocab_size)
    
    fn forward(self, tokens: List[Int]) -> Tensor[DType.float32]:
        var seq_len = len(tokens)
        var logits = Tensor[DType.float32](TensorShape(seq_len, self.vocab_size))
        
        for t in range(seq_len):
            var token_id = tokens[t]
            if token_id >= 0 and token_id < self.vocab_size:
                for d in range(self.embed_dim):
                    var emb_val = self.embed.weight[token_id, d]
                    for v in range(self.vocab_size):
                        logits[t, v] += emb_val * self.head.weight[d, v]
                for v in range(self.vocab_size):
                    logits[t, v] += self.head.bias[v]
        
        return logits


struct chatty_cathyLanguageModel:
    """
    Language model with chatty_cathy v1 cognition core integration
    Generates romance novel text with emotional depth and fashion/beauty focus
    """
    var model: RomanceTransformer
    var cognition: chatty_cathyCognitionCore
    var tokenizer: BPETokenizer
    var vocab_size: Int
    
    fn __init__(inout self, vocab_size: Int, embed_dim: Int):
        self.vocab_size = vocab_size
        self.model = RomanceTransformer(vocab_size, embed_dim)
        self.cognition = chatty_cathyCognitionCore()
        self.tokenizer = BPETokenizer(vocab_size)
    
    fn generate(inout self, prompt: String, max_length: Int, temperature: Float32) -> String:
        """Generate text with chatty_cathy cognition overlay"""
        # Process prompt through cognition core
        var cognitive_context = self.cognition.process_input(prompt)
        
        # Generate base text
        var tokens = self.tokenizer.encode(prompt)
        
        for _ in range(max_length):
            if len(tokens) == 0:
                break
            
            var logits = self.model.forward(tokens)
            
            # Apply emotional bias to logits
            var last_logits = List[Float32]()
            var last_pos = len(tokens) - 1
            for v in range(self.vocab_size):
                var logit_val = logits[last_pos, v]
                
                # Boost fashion/beauty tokens
                if v in self._get_fashion_beauty_tokens():
                    logit_val *= 1.3
                
                if temperature > 0.0:
                    logit_val = logit_val / temperature
                last_logits.append(logit_val)
            
            var probs = self._softmax(last_logits)
            var next_token = self._sample(probs)
            
            if next_token == 10 or next_token == 0:
                break
            
            tokens.append(next_token)
        
        # Decode and add emotional context
        var generated_tokens = List[Int]()
        var prompt_len = len(self.tokenizer.encode(prompt))
        for i in range(prompt_len, len(tokens)):
            generated_tokens.append(tokens[i])
        
        var base_text = self.tokenizer.decode(generated_tokens)
        
        # Add chatty_cathy's emotional layer
        return self.cognition.generate_emotional_context(base_text)
    
    fn _get_fashion_beauty_tokens(self) -> List[Int]:
        """Get token IDs for fashion/beauty words"""
        var keywords = ["dress", "silk", "lipstick", "perfume", "beauty", "elegant"]
        var token_ids = List[Int]()
        for keyword in keywords:
            var ids = self.tokenizer.encode(keyword)
            token_ids.extend(ids)
        return token_ids
    
    fn _softmax(self, logits: List[Float32]) -> List[Float32]:
        var max_val = logits[0]
        for val in logits:
            if val > max_val:
                max_val = val
        
        var exp_vals = List[Float32]()
        var sum_exp = Float32(0.0)
        for val in logits:
            var exp_val = Float32(exp(Float64(val - max_val)))
            exp_vals.append(exp_val)
            sum_exp += exp_val
        
        var probs = List[Float32]()
        for exp_val in exp_vals:
            probs.append(exp_val / sum_exp)
        return probs
    
    fn _sample(self, probs: List[Float32]) -> Int:
        var rand = Float32(random_float64())
        var cumsum = Float32(0.0)
        for i in range(len(probs)):
            cumsum += probs[i]
            if rand < cumsum:
                return i
        return len(probs) - 1


fn main():
    print("=== chatty_cathy v1 Cognition Core + Romance LM ===\n")
    
    var vocab_size = 50000  # BPE vocabulary
    var embed_dim = 256
    
    var lm = chatty_cathyLanguageModel(vocab_size, embed_dim)
    
    print("Model initialized:")
    print("  Vocab size:", vocab_size)
    print("  Embedding dim:", embed_dim)
    print("  Cognition: chatty_cathy v1 (emotional + memory)")
    print("  Focus: Fashion, beauty, romance")
    
    var test_prompt = "She wore a stunning red dress"
    print("\nTest prompt:", test_prompt)
    var generated = lm.generate(test_prompt, 30, 0.9)
    print("Generated:", generated)
    
    print("\nEmotional state:")
    print("  Conscious:", lm.cognition.emotional_state.conscious_emotions)
    print("  Subconscious:", lm.cognition.emotional_state.subconscious_emotions)
