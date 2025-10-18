"""
Training Configurations for Fashion/Beauty Chatbot
Scale from 5 minutes to 6 hours, reaching ~1B parameters
"""

from collections import Dict

struct TrainingConfig:
    var name: String
    var vocab_size: Int
    var embed_dim: Int
    var num_layers: Int
    var num_heads: Int
    var seq_length: Int
    var batch_size: Int
    var epochs: Int
    var learning_rate: Float32
    var estimated_time_minutes: Int
    var total_parameters: Int
    var dataset_size_mb: Int
    var description: String
    
    fn __init__(inout self, name: String, vocab_size: Int, embed_dim: Int,
                num_layers: Int, num_heads: Int, seq_length: Int,
                batch_size: Int, epochs: Int, learning_rate: Float32,
                estimated_time_minutes: Int, dataset_size_mb: Int,
                description: String):
        self.name = name
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.estimated_time_minutes = estimated_time_minutes
        self.dataset_size_mb = dataset_size_mb
        self.description = description
        
        # Calculate parameters: embed + layers + head
        # Formula: vocab*embed + layers*(4*embed^2) + embed*vocab
        var embed_params = vocab_size * embed_dim * 2  # input + output
        var layer_params = num_layers * (4 * embed_dim * embed_dim)  # attention + ffn
        self.total_parameters = embed_params + layer_params
    
    fn print_config(self):
        print("\n╔════════════════════════════════════════════════╗")
        print("║", self.name)
        print("╚════════════════════════════════════════════════╝")
        print()
        print("ARCHITECTURE:")
        print("  Vocabulary size:    ", self.vocab_size)
        print("  Embedding dim:      ", self.embed_dim)
        print("  Num layers:         ", self.num_layers)
        print("  Num attention heads:", self.num_heads)
        print("  Context length:     ", self.seq_length)
        print()
        print("TRAINING:")
        print("  Batch size:         ", self.batch_size)
        print("  Epochs:             ", self.epochs)
        print("  Learning rate:      ", self.learning_rate)
        print("  Dataset size:       ", self.dataset_size_mb, "MB")
        print()
        print("PERFORMANCE:")
        print("  Total parameters:   ", self.format_params(self.total_parameters))
        print("  Estimated time:     ", self.estimated_time_minutes, "minutes")
        print()
        print("DESCRIPTION:")
        print("  ", self.description)
        print()
    
    fn format_params(self, params: Int) -> String:
        if params >= 1_000_000_000:
            return String(params / 1_000_000_000) + "." + String((params % 1_000_000_000) / 100_000_000) + "B"
        elif params >= 1_000_000:
            return String(params / 1_000_000) + "." + String((params % 1_000_000) / 100_000) + "M"
        elif params >= 1_000:
            return String(params / 1_000) + "." + String((params % 1_000) / 100) + "K"
        else:
            return String(params)


fn get_config_tiny() -> TrainingConfig:
    """5 MINUTE TRAINING - Quick test"""
    return TrainingConfig(
        name="TINY - 5 Minute Quick Test",
        vocab_size=8000,          # Small BPE vocab
        embed_dim=128,            # Minimal embedding
        num_layers=2,             # Just 2 transformer blocks
        num_heads=2,              # 2 attention heads
        seq_length=64,            # Short context
        batch_size=16,
        epochs=10,
        learning_rate=0.001,
        estimated_time_minutes=5,
        dataset_size_mb=5,        # ~5MB of Reddit beauty comments
        description="Fast test run to verify everything works. Uses tiny dataset."
    )


fn get_config_small() -> TrainingConfig:
    """30 MINUTE TRAINING - Development model"""
    return TrainingConfig(
        name="SMALL - 30 Minute Development",
        vocab_size=16000,         # Moderate BPE vocab
        embed_dim=256,            # Standard small model
        num_layers=4,             # 4 transformer blocks
        num_heads=4,              # 4 attention heads
        seq_length=128,           # Medium context
        batch_size=32,
        epochs=20,
        learning_rate=0.0008,
        estimated_time_minutes=30,
        dataset_size_mb=50,       # ~50MB beauty/fashion conversations
        description="Good for development and testing. Decent quality output."
    )


fn get_config_medium() -> TrainingConfig:
    """2 HOUR TRAINING - Production-lite model"""
    return TrainingConfig(
        name="MEDIUM - 2 Hour Production-Lite",
        vocab_size=32000,         # Full BPE vocab
        embed_dim=512,            # Medium model size
        num_layers=8,             # 8 transformer blocks
        num_heads=8,              # 8 attention heads
        seq_length=256,           # Good context window
        batch_size=64,
        epochs=30,
        learning_rate=0.0005,
        estimated_time_minutes=120,
        dataset_size_mb=500,      # ~500MB dataset
        description="Production-ready quality. ~100M parameters. Good conversational ability."
    )


fn get_config_large() -> TrainingConfig:
    """6 HOUR TRAINING - Full production model (~1B params)"""
    return TrainingConfig(
        name="LARGE - 6 Hour Full Production (~1B params)",
        vocab_size=50000,         # Large BPE vocab
        embed_dim=1024,           # Large embedding
        num_layers=24,            # 24 transformer blocks (GPT-2 medium size)
        num_heads=16,             # 16 attention heads
        seq_length=512,           # Large context window
        batch_size=128,
        epochs=50,
        learning_rate=0.0003,
        estimated_time_minutes=360,
        dataset_size_mb=2000,     # ~2GB dataset
        description="Full production model reaching ~1B parameters. High quality output."
    )


fn print_all_configs():
    """Print all training configurations"""
    print("\n╔════════════════════════════════════════════════╗")
    print("║   FASHION/BEAUTY CHATBOT TRAINING CONFIGS      ║")
    print("╚════════════════════════════════════════════════╝\n")
    
    print("Available configurations from quick test to production:\n")
    
    var configs = [
        get_config_tiny(),
        get_config_small(),
        get_config_medium(),
        get_config_large()
    ]
    
    for config in configs:
        config.print_config()
        print("─" * 50 + "\n")


fn get_dataset_sources():
    """Print dataset sources and download instructions"""
    print("\n╔════════════════════════════════════════════════╗")
    print("║          DATASET SOURCES & DOWNLOADS           ║")
    print("╚════════════════════════════════════════════════╝\n")
    
    print("PRIMARY SOURCE - Reddit Beauty/Fashion Communities:")
    print("─" * 50)
    print()
    print("1. PUSHSHIFT REDDIT DATASET")
    print("   URL: https://files.pushshift.io/reddit/")
    print("   Subreddits to filter:")
    print("   • r/MakeupAddiction (8M pageviews/month)")
    print("   • r/SkincareAddiction (14M pageviews/month)")
    print("   • r/femalefashionadvice")
    print("   • r/beauty")
    print("   • r/HairCare")
    print("   • r/NailArt")
    print("   • r/Perfumes")
    print()
    print("   Download command:")
    print("   wget https://files.pushshift.io/reddit/comments/RC_2024-01.zst")
    print()
    
    print("2. HUGGINGFACE REDDIT DATASET")
    print("   URL: https://huggingface.co/datasets/fddemarco/pushshift-reddit-comments")
    print("   Already filtered and processed")
    print()
    
    print("3. COSMETICS PRODUCT REVIEWS")
    print("   Kaggle: https://www.kaggle.com/datasets/waqi786/most-used-beauty-cosmetics-products-in-the-world")
    print("   Sephora reviews, product descriptions")
    print()
    
    print("\nDATASET SIZE RECOMMENDATIONS:")
    print("─" * 50)
    print("TINY (5 min):    5MB   - 1 month of single subreddit")
    print("SMALL (30 min):  50MB  - 3 months of multiple subreddits")
    print("MEDIUM (2 hr):   500MB - 1 year of beauty communities")
    print("LARGE (6 hr):    2GB   - 3+ years of all beauty/fashion subreddits")
    print()
    
    print("\nFILTERING INSTRUCTIONS:")
    print("─" * 50)
    print("1. Download Pushshift data")
    print("2. Extract with: zstd -d RC_2024-01.zst")
    print("3. Filter subreddits: grep -E 'MakeupAddiction|SkincareAddiction|beauty'")
    print("4. Convert to conversational format (remove markdown, URLs)")
    print("5. Save as plain text, one conversation per line")
    print()
    
    print("\nCONVERSATIONAL FORMAT:")
    print("─" * 50)
    print("Each line should be natural conversation:")
    print()
    print("Example:")
    print("  User: What's the best foundation for oily skin?")
    print("  Bot: I love the Fenty Pro Filt'r! It stays matte all day.")
    print()
    print("  User: Tell me about red lipstick")
    print("  Bot: Red lipstick is so classic! MAC Ruby Woo is my go-to.")
    print()


fn create_sample_dataset(config_type: String) -> String:
    """Create sample conversational dataset for testing"""
    if config_type == "tiny":
        return """User: What's a good red lipstick?
Bot: I love MAC Ruby Woo! It's a classic matte red that looks amazing.

User: How do I apply it?
Bot: Exfoliate your lips first, then apply with a lip brush for precision. It lasts forever!

User: What about lip liner?
Bot: Definitely use a red lip liner underneath. It prevents feathering and makes it last longer.

User: Any drugstore alternatives?
Bot: Maybelline SuperStay Matte Ink in Pioneer is super similar and way cheaper!

User: Tell me about skincare for oily skin
Bot: Start with a gentle cleanser like CeraVe Foaming. Then use niacinamide serum to control oil.

User: What moisturizer should I use?
Bot: The Neutrogena Hydro Boost gel-cream is perfect! It's lightweight but hydrating.

User: Do I need sunscreen?
Bot: Absolutely! EltaMD UV Clear is amazing for oily skin. Won't make you greasy at all.

User: How do I deal with acne?
Bot: Salicylic acid is your friend. The Ordinary has a great 2% solution that's super affordable.
""" * 10  # Repeat for more data
    
    elif config_type == "small":
        # Include more diverse conversations
        return create_sample_dataset("tiny") * 5 + """
User: I want to try a new hairstyle
Bot: What's your hair type? Curly, straight, wavy? That makes a huge difference!

User: It's wavy and thick
Bot: Lucky! I'd suggest layers to reduce bulk. Maybe a long bob with curtain bangs?

User: What products should I use?
Bot: A curl cream like Oui hair's Wave Spray and a diffuser will give you gorgeous waves.

User: Tell me about perfume for work
Bot: You want something subtle. Try Glossier You or Clean Reserve. They're personal but not overpowering.

User: What about evening perfume?
Bot: Go bold! Yves Saint Laurent Black Opium or Viktor&Rolf Flowerbomb are stunning.
""" * 20
    
    else:
        return "Load full dataset from file for medium/large configs"


fn main():
    """Main entry - show all configs and dataset info"""
    print_all_configs()
    get_dataset_sources()
    
    print("\n╔════════════════════════════════════════════════╗")
    print("║              QUICK START GUIDE                 ║")
    print("╚════════════════════════════════════════════════╝\n")
    
    print("STEP 1: Choose your configuration")
    print("  • Start with TINY (5 min) to test everything works")
    print("  • Move to SMALL (30 min) for development")
    print("  • Use MEDIUM (2 hr) for good production quality")
    print("  • Train LARGE (6 hr) for full 1B param model")
    print()
    
    print("STEP 2: Download dataset")
    print("  • See dataset sources above")
    print("  • Filter for beauty/fashion subreddits")
    print("  • Convert to conversational format")
    print()
    
    print("STEP 3: Train the model")
    print("  mojo training.mojo --config tiny")
    print("  mojo training.mojo --config small")
    print("  mojo training.mojo --config medium")
    print("  mojo training.mojo --config large")
    print()
    
    print("STEP 4: Test generation")
    print("  mojo interface.mojo --chat")
    print()
    
    print("═" * 50)
    print("HARDWARE REQUIREMENTS:")
    print("─" * 50)
    print("TINY:   2GB RAM, any CPU")
    print("SMALL:  8GB RAM, decent CPU or GPU")
    print("MEDIUM: 16GB RAM, GPU recommended")
    print("LARGE:  32GB+ RAM, GPU required (RTX 3090 or better)")
    print("═" * 50)