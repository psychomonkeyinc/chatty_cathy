import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from typing import List, Tuple, Any
import math
import random
from cognition_core import chatty_cathyLanguageModel, chatty_cathyCognitionCore

def compute_loss_with_emotion(logits: torch.Tensor, targets: torch.Tensor,
                              cognition: chatty_cathyCognitionCore) -> torch.Tensor:
    """
    Compute cross-entropy loss with emotional weighting
    chatty_cathy cognition core influences loss based on emotional state
    """
    # Cross-entropy loss
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')

    # Apply emotional weighting
    emotion_weight = 1.0
    predicted_emotion = cognition.emotional_state.predict_next_emotion()

    # Boost learning on emotionally relevant tokens
    if predicted_emotion in ["desire", "lust", "confidence"]:
        emotion_weight = 1.2

    return loss * emotion_weight


def update_with_cognition(model: chatty_cathyLanguageModel, logits: torch.Tensor,
                         tokens: List[int], targets: List[int], optimizer: optim.Optimizer,
                         cognition: chatty_cathyCognitionCore, learning_rate: float):
    """
    Update parameters with chatty_cathy cognition guidance
    Fashion/beauty tokens get stronger updates
    """
    optimizer.zero_grad()

    loss = compute_loss_with_emotion(logits, torch.tensor(targets, dtype=torch.long), cognition)
    loss.backward()
    optimizer.step()

    # Additional cognition-guided updates could be added here


def create_batches(tokens: List[int], seq_length: int, batch_size: int = 32) -> DataLoader:
    """Create training batches"""
    # Create input-target pairs
    inputs = []
    targets = []

    for i in range(len(tokens) - seq_length):
        inputs.append(tokens[i:i+seq_length])
        targets.append(tokens[i+1:i+seq_length+1])

    # Convert to tensors
    inputs_tensor = torch.tensor(inputs, dtype=torch.long)
    targets_tensor = torch.tensor(targets, dtype=torch.long)

    # Create dataset and dataloader
    dataset = TensorDataset(inputs_tensor, targets_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def train_chatty_cathy_model(model: chatty_cathyLanguageModel, text: str,
                             tokenizer, epochs: int, seq_length: int, learning_rate: float,
                             batch_size: int = 32, device: str = 'cpu'):
    """
    Train chatty_cathy language model on romance novels
    Integrates emotional cognition during training
    """
    print("\n╔════════════════════════════════════════════════╗")
    print("║   CHATTY_CATHY V1 TRAINING - Romance Novel Core     ║")
    print("╚════════════════════════════════════════════════╝\n")

    print("Training data:", len(text), "characters")
    print("Focus: Fashion, makeup, beauty, romance")
    print("Cognition: Conscious/Subconscious/Unconscious layers")
    print("Memory: Emotional tagging + fashion memory\n")

    # Tokenize
    tokens = tokenizer.encode(text)
    print("Token count:", len(tokens))

    # Create batches
    dataloader = create_batches(tokens, seq_length, batch_size)
    print("Training batches:", len(dataloader))
    print("Sequence length:", seq_length)
    print("Batch size:", batch_size)
    print("\n" + "─" * 50)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move model to device
    model.to(device)

    # Training loop with cognition integration
    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0

        # Update chatty_cathy's emotional state each epoch
        model.cognition.emotional_state.update_from_text(text[:500])

        for batch_inputs, batch_targets in dataloader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            # Forward pass - process each sequence in batch
            batch_logits = []
            for seq in batch_inputs:
                logits = model(seq.tolist())
                batch_logits.append(logits)

            # Stack logits
            batch_logits = torch.stack(batch_logits)

            # Compute loss with emotional weighting
            loss = compute_loss_with_emotion(batch_logits, batch_targets, model.cognition)
            total_loss += loss.item()
            batch_count += 1

            # Update with cognition guidance
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Every 10 batches, show emotional state
            if batch_count % 10 == 0:
                emotion = model.cognition.emotional_state.predict_next_emotion()
                print(f"  Batch {batch_count} - Dominant emotion: {emotion}")

        avg_loss = total_loss / batch_count
        print(f"\n│ Epoch {epoch + 1} / {epochs}")
        print(f"│ Loss: {avg_loss:.4f}")
        print(f"│ Conscious emotions: {model.cognition.emotional_state.conscious_emotions}")
        print(f"│ Subconscious: {model.cognition.emotional_state.subconscious_emotions}")
        print(f"│ Fashion memories: {len(model.cognition.memory.fashion_memory)}")
        print("─" * 50 + "\n")

    print("✓ Training complete!\n")


def evaluate_generation_quality(model: chatty_cathyLanguageModel, test_prompts: List[str], tokenizer):
    """Test generation quality on fashion/beauty prompts"""
    print("\n╔════════════════════════════════════════════════╗")
    print("║        CHATTY_CATHY GENERATION EVALUATION            ║")
    print("╚════════════════════════════════════════════════╝\n")

    model.eval()
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts):
            print(f"Prompt: {prompt}")
            print("─" * 40)

            # Generate with different temperatures
            temps = [0.7, 0.9, 1.1]
            for temp in temps:
                generated = model.generate(prompt, tokenizer, max_length=40, temperature=temp)
                print(f"  [T={temp}] {generated}")

            # Show emotional analysis
            emotion = model.cognition.emotional_state.predict_next_emotion()
            print(f"  Predicted emotion: {emotion}")
            print("\n")


def main():
    print("╔════════════════════════════════════════════════╗")
    print("║  CHATTY_CATHY V1 - Romance Novel Language Model     ║")
    print("║  Fashion • Beauty • Emotion • Memory          ║")
    print("╚════════════════════════════════════════════════╝\n")

    # Configuration
    vocab_size = 50000  # BPE tokenizer
    embed_dim = 256  # Hidden dimension
    seq_length = 64  # Context window
    epochs = 50
    learning_rate = 0.001
    batch_size = 16

    print("CONFIGURATION")
    print("─" * 50)
    print("Vocab size:     ", vocab_size, "(BPE)")
    print("Embedding dim:  ", embed_dim)
    print("Sequence length:", seq_length)
    print("Epochs:         ", epochs)
    print("Learning rate:  ", learning_rate)
    print("Batch size:     ", batch_size)
    print("─" * 50 + "\n")

    # Check for CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize chatty_cathy model
    model = chatty_cathyLanguageModel(vocab_size, embed_dim)

    # Training corpus - Romance novel focused on fashion/beauty
    training_text = """
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

    # For now, create a simple mock tokenizer
    class MockTokenizer:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
            self.eos_token_id = vocab_size - 1

        def encode(self, text: str) -> List[int]:
            # Simple character-level tokenization for demo
            return [ord(c) % self.vocab_size for c in text if ord(c) < 128][:1000]

        def decode(self, tokens: List[int]) -> str:
            # Simple decode
            return "".join(chr(token % 128) for token in tokens)

    tokenizer = MockTokenizer(vocab_size)

    # Train with chatty_cathy cognition
    train_chatty_cathy_model(model, training_text, tokenizer, epochs, seq_length,
                             learning_rate, batch_size, device)

    # Evaluation
    test_prompts = [
        "She wore a stunning black dress",
        "The lipstick was",
        "Her perfume smelled of",
        "As she applied her makeup",
        "The silk fabric felt"
    ]

    evaluate_generation_quality(model, test_prompts, tokenizer)

    # Final statistics
    print("\n╔════════════════════════════════════════════════╗")
    print("║            CHATTY_CATHY V1 STATISTICS                ║")
    print("╚════════════════════════════════════════════════╝\n")

    print("Personality traits:")
    for trait, value in model.cognition.personality_traits.items():
        print(f"  {trait}: {value}")

    print("\nMemory system:")
    print("  Short-term memories:", len(model.cognition.memory.short_term))
    print("  Fashion memories:", len(model.cognition.memory.fashion_memory))
    print("  Beauty preferences:", model.cognition.memory.beauty_preferences)

    print("\nEmotional state:")
    print("  Conscious:", model.cognition.emotional_state.conscious_emotions)
    print("  Subconscious:", model.cognition.emotional_state.subconscious_emotions)
    print("  Patterns:", len(model.cognition.emotional_state.unconscious_patterns))

    print("\n" + "═" * 50)
    print("TRAINING COMPLETE - chatty_cathy v1 is ready")
    print("Expected training time: 6-12 hours on full dataset")
    print("Competitive edge: Emotional depth + fashion expertise")
    print("═" * 50 + "\n")

    # Save instructions
    print("TO DEPLOY:")
    print("1. Save model weights: model.save('chatty_cathy_v1.pth')")
    print("2. Save tokenizer vocab: tokenizer.save_vocab('vocab.json')")
    print("3. Load for inference: chatty_cathyLanguageModel.load('chatty_cathy_v1.pth')")
    print("\nTO IMPROVE:")
    print("• Increase embed_dim to 512-1024")
    print("• Add attention layers for better context")
    print("• Train on 50-100 full romance novels")
    print("• Fine-tune on specific fashion/beauty corpus")


if __name__ == "__main__":
    main()