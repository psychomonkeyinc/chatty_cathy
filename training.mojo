from tensor import Tensor, TensorShape
from collections import List
from math import log, exp
from random import random_float64
from input_token_output import TokenIO
from cognition_core import chatty_cathyLanguageModel, chatty_cathyCognitionCore

fn compute_loss_with_emotion(logits: Tensor[DType.float32], targets: List[Int],
                              cognition: chatty_cathyCognitionCore) -> Float32:
    """
    Compute cross-entropy loss with emotional weighting
    chatty_cathy cognition core influences loss based on emotional state
    """
    var total_loss = Float32(0.0)
    var seq_len = len(targets)
    
    for t in range(seq_len):
        var target_id = targets[t]
        if target_id < 0 or target_id >= logits.shape()[1]:
            continue
        
        var timestep_logits = List[Float32]()
        for v in range(logits.shape()[1]):
            timestep_logits.append(logits[t, v])
        
        var probs = softmax_helper(timestep_logits)
        var target_prob = probs[target_id]
        
        if target_prob > 1e-10:
            var loss = -Float32(log(Float64(target_prob)))
            
            # Apply emotional weighting
            var emotion_weight = Float32(1.0)
            var predicted_emotion = cognition.emotional_state.predict_next_emotion()
            
            # Boost learning on emotionally relevant tokens
            if predicted_emotion in ["desire", "lust", "confidence"]:
                emotion_weight = 1.2
            
            total_loss += loss * emotion_weight
        else:
            total_loss += Float32(10.0)
    
    return total_loss / Float32(seq_len)


fn update_with_cognition(inout model: chatty_cathyLanguageModel, logits: Tensor[DType.float32],
                         tokens: List[Int], targets: List[Int], learning_rate: Float32):
    """
    Update parameters with chatty_cathy cognition guidance
    Fashion/beauty tokens get stronger updates
    """
    var seq_len = len(targets)
    var vocab_size = model.vocab_size
    var embed_dim = model.model.embed_dim
    
    # Get fashion/beauty token set for boosted learning
    var fashion_tokens = model._get_fashion_beauty_tokens()
    
    for t in range(seq_len):
        var target_id = targets[t]
        if target_id < 0 or target_id >= vocab_size:
            continue
        
        var token_id = tokens[t]
        if token_id < 0 or token_id >= vocab_size:
            continue
        
        var timestep_logits = List[Float32]()
        for v in range(vocab_size):
            timestep_logits.append(logits[t, v])
        var probs = softmax_helper(timestep_logits)
        
        for v in range(vocab_size):
            var grad = probs[v]
            if v == target_id:
                grad -= 1.0
            
            # Boost learning rate for fashion/beauty tokens
            var effective_lr = learning_rate
            if v in fashion_tokens:
                effective_lr *= 1.5
            
            # Update model parameters
            for d in range(embed_dim):
                var emb_val = model.model.embed.weight[token_id, d]
                model.model.head.weight[d, v] -= effective_lr * grad * emb_val
            
            model.model.head.bias[v] -= effective_lr * grad
            
            for d in range(embed_dim):
                var weight_val = model.model.head.weight[d, v]
                model.model.embed.weight[token_id, d] -= effective_lr * grad * weight_val


fn create_batches(tokens: List[Int], seq_length: Int) -> (List[List[Int]], List[List[Int]]):
    """Create training batches"""
    var inputs = List[List[Int]]()
    var targets = List[List[Int]]()
    
    var num_sequences = len(tokens) // seq_length
    
    for i in range(num_sequences):
        var input_seq = List[Int]()
        var target_seq = List[Int]()
        
        for j in range(seq_length):
            var idx = i * seq_length + j
            if idx + 1 < len(tokens):
                input_seq.append(tokens[idx])
                target_seq.append(tokens[idx + 1])
        
        if len(input_seq) == seq_length:
            inputs.append(input_seq)
            targets.append(target_seq)
    
    return (inputs, targets)


fn softmax_helper(logits: List[Float32]) -> List[Float32]:
    """Helper softmax function"""
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


fn train_chatty_cathy_model(inout model: chatty_cathyLanguageModel, text: String,
                      epochs: Int, seq_length: Int, learning_rate: Float32):
    """
    Train chatty_cathy language model on romance novels
    Integrates emotional cognition during training
    """
    print("\n╔════════════════════════════════════════════════╗")
    print("║   chatty_cathy v1 TRAINING - Romance Novel Core     ║")
    print("╚════════════════════════════════════════════════╝\n")
    
    print("Training data:", len(text), "characters")
    print("Focus: Fashion, makeup, beauty, romance")
    print("Cognition: cognition_core/Subcognition_core/Uncognition_core layers")
    print("Memory: Emotional tagging + fashion memory\n")
    
    # Tokenize
    var tokens = model.tokenizer.encode(text)
    print("Token count:", len(tokens))
    
    # Create batches
    var batches = create_batches(tokens, seq_length)
    var inputs = batches[0]
    var targets = batches[1]
    print("Training batches:", len(inputs))
    print("Sequence length:", seq_length)
    print("\n" + "─" * 50)
    
    # Training loop with cognition integration
    for epoch in range(epochs):
        var total_loss = Float32(0.0)
        var batch_count = 0
        
        # Update chatty_cathy's emotional state each epoch
        model.cognition.emotional_state.update_from_text(text[:500])
        
        for b in range(len(inputs)):
            # Forward pass
            var logits = model.model.forward(inputs[b])
            
            # Compute loss with emotional weighting
            var loss = compute_loss_with_emotion(logits, targets[b], model.cognition)
            total_loss += loss
            batch_count += 1
            
            # Update with cognition guidance
            update_with_cognition(model, logits, inputs[b], targets[b], learning_rate)
            
            # Every 10 batches, show emotional state
            if b % 10 == 0 and b > 0:
                var emotion = model.cognition.emotional_state.predict_next_emotion()
                print("  Batch", b, "- Dominant emotion:", emotion)
        
        var avg_loss = total_loss / Float32(batch_count)
        print("\n│ Epoch", epoch + 1, "/", epochs)
        print("│ Loss:", avg_loss)
        print("│ cognition_core emotions:", model.cognition.emotional_state.cognition_core_emotions)
        print("│ Subcognition_core:", model.cognition.emotional_state.subcognition_core_emotions)
        print("│ Fashion memories:", len(model.cognition.memory.fashion_memory))
        print("─" * 50 + "\n")
    
    print("✓ Training complete!\n")


fn evaluate_generation_quality(inout model: chatty_cathyLanguageModel, test_prompts: List[String]):
    """Test generation quality on fashion/beauty prompts"""
    print("\n╔════════════════════════════════════════════════╗")
    print("║        chatty_cathy GENERATION EVALUATION            ║")
    print("╚════════════════════════════════════════════════╝\n")
    
    for i in range(len(test_prompts)):
        var prompt = test_prompts[i]
        print("Prompt:", prompt)
        print("─" * 40)
        
        # Generate with different temperatures
        var temps = [0.7, 0.9, 1.1]
        for temp in temps:
            var generated = model.generate(prompt, 40, temp)
            print(f"  [T={temp}] {generated}")
        
        # Show emotional analysis
        var emotion = model.cognition.emotional_state.predict_next_emotion()
        print("  Predicted emotion:", emotion)
        print("\n")


fn main():
    print("╔════════════════════════════════════════════════╗")
    print("║  chatty_cathy v1 - Romance Novel Language Model     ║")
    print("║  Fashion • Beauty • Emotion • Memory          ║")
    print("╚════════════════════════════════════════════════╝\n")
    
    # Configuration
    var vocab_size = 50000  # BPE tokenizer
    var embed_dim = 256  # Hidden dimension
    var seq_length = 64  # Context window
    var epochs = 50
    var learning_rate = Float32(0.001)
    
    print("CONFIGURATION")
    print("─" * 50)
    print("Vocab size:     ", vocab_size, "(BPE)")
    print("Embedding dim:  ", embed_dim)
    print("Sequence length:", seq_length)
    print("Epochs:         ", epochs)
    print("Learning rate:  ", learning_rate)
    print("─" * 50 + "\n")
    
    # Initialize chatty_cathy model
    var model = chatty_cathyLanguageModel(vocab_size, embed_dim)
    
    # Training corpus - Romance novel focused on fashion/beauty
    var training_text = """
    She stood before the mirror, the crimson silk dress clinging to her curves 
    like a second skin. The fabric was exquisite, imported from Paris, and worth 
    every penny. Her makeup was flawless - smoky eyes that smoldered with barely 
    contained desire, lips painted a deep burgundy that matched the dress perfectly.
    
    The perfume she'd chosen was intoxicating, a custom blend of jasmine, vanilla, 
    and something darker, more sensual. It left a trail wherever she walked, a 
    whisper of mystery and promise. Her hair cascaded down her back in loose waves, 
    each strand catching the candlelight like spun gold.
    
    She felt powerful. Beautiful. Dangerous.
    
    The diamond necklace at her throat was a gift from her lover, though she'd 
    never admit how much she treasured it. The stones caught the light with every 
    breath, every subtle movement. She turned slowly, admiring herself from every 
    angle, noting how the dress accentuated her figure, how the makeup transformed 
    her into someone new, someone bold.
    
    Tonight would change everything. She could feel it in the way her heart raced, 
    in the flutter of anticipation low in her belly. The black heels she wore added 
    four inches to her height, making her legs look endless. The stockings were 
    sheer silk, a touch of elegance that only she would know about.
    
    Her lipstick was perfect, her eyeliner sharp enough to cut. She was a vision 
    of feminine power, wrapped in designer fabric and expensive perfume. The pearls 
    in her ears were vintage, passed down from her grandmother, a reminder that 
    beauty and strength ran in her blood.
    
    She grabbed her clutch, a tiny thing studded with crystals that caught the 
    light like stars. Inside was her lipstick for touch-ups, her phone, and a 
    small vial of perfume. Everything she needed to maintain the illusion of 
    effortless beauty.
    
    The lace of her lingerie was French, delicate and expensive, hidden beneath 
    the silk dress like a secret. She smiled at her reflection, knowing that 
    confidence was the best accessory. The mascara made her lashes impossibly long, 
    framing her eyes in a way that was both innocent and seductive.
    
    She was ready. The night awaited, full of possibility and promise.
    """ * 20  # Repeat for more training data
    
    # DATASET INSTRUCTIONS
    print("DATASET SOURCES")
    print("─" * 50)
    print("1. Goodreads Romance Dataset:")
    print("   → https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html")
    print("   → 335,449 romance books + 3.5M reviews")
    print()
    print("2. Project Gutenberg Romance:")
    print("   → https://www.gutenberg.org/ebooks/subject/2487")
    print("   → Filter for fashion/beauty keywords")
    print()
    print("3. BookCorpus Romance Subset:")
    print("   → 11,038 books including romance")
    print("─" * 50 + "\n")
    
    print("PREPROCESSING STEPS:")
    print("1. Download dataset")
    print("2. Filter for fashion/beauty/risque content")
    print("3. Train BPE tokenizer (vocab_size=50k)")
    print("4. Run training with chatty_cathy cognition core\n")
    
    # Train with chatty_cathy cognition
    train_chatty_cathy_model(model, training_text, epochs, seq_length, learning_rate)
    
    # Evaluation
    var test_prompts = List[String]()
    test_prompts.append("She wore a stunning black dress")
    test_prompts.append("The lipstick was")
    test_prompts.append("Her perfume smelled of")
    test_prompts.append("As she applied her makeup")
    test_prompts.append("The silk fabric felt")
    
    evaluate_generation_quality(model, test_prompts)
    
    # Final statistics
    print("\n╔════════════════════════════════════════════════╗")
    print("║            chatty_cathy v1 STATISTICS                ║")
    print("╚════════════════════════════════════════════════╝\n")
    
    print("Personality traits:")
    for trait, value in model.cognition.personality_traits.items():
        print(f"  {trait}: {value}")
    
    print("\nMemory system:")
    print("  Short-term memories:", len(model.cognition.memory.short_term))
    print("  Fashion memories:", len(model.cognition.memory.fashion_memory))
    print("  Beauty preferences:", model.cognition.memory.beauty_preferences)
    
    print("\nEmotional state:")
    print("  cognition_core:", model.cognition.emotional_state.cognition_core_emotions)
    print("  Subcognition_core:", model.cognition.emotional_state.subcognition_core_emotions)
    print("  Patterns:", len(model.cognition.emotional_state.uncognition_core_patterns))
    
    print("\n" + "═" * 50)
    print("TRAINING COMPLETE - chatty_cathy v1 is ready")
    print("Expected training time: 6-12 hours on full dataset")
    print("Competitive edge: Emotional depth + fashion expertise")
    print("═" * 50 + "\n")
    
    # Save instructions
    print("TO DEPLOY:")
    print("1. Save model weights: model.save('chatty_cathy_v1.mojo')")
    print("2. Save tokenizer vocab: tokenizer.save_vocab('vocab.json')")
    print("3. Load for inference: chatty_cathyLanguageModel.load('chatty_cathy_v1.mojo')")
    print("\nTO IMPROVE:")
    print("• Increase embed_dim to 512-1024")
    print("• Add attention layers for better context")
    print("• Train on 50-100 full romance novels")
    print("• Fine-tune on specific fashion/beauty corpus")
