# chatty_cathy v4 - Fashion/Beauty Romance Language Model

A sophisticated AI chatbot specializing in fashion, beauty, and romance conversations, built with Mojo programming language.

## Features

- **Emotional Intelligence**: Multi-layer cognition with conscious/subconscious/unconscious processing
- **Fashion Expertise**: Specialized knowledge of clothing, makeup, and beauty products
- **Memory Systems**: 5-tier memory architecture for long-term learning
- **Romance Focus**: Trained on romance novel datasets with emotional depth
- **Interactive Interface**: Chat, generation, and emotion-aware responses

## Architecture

### Core Components

- `chatty_cathy.mojo` - Main language model with emotional cognition
- `training.mojo` - Training logic with emotional weighting
- `interface.mojo` - User interface for chat and generation
- `tokenizer.mojo` - BPE tokenizer optimized for romance text
- `chatty_cathy_meta_controller.mojo` - Advanced meta-reasoning and memory consolidation

### Key Features

- **Emotional State Tracking**: Conscious, subconscious, and unconscious emotional layers
- **Memory Tiers**: Working, episodic, semantic, emotional, and knowledge memory
- **Fashion Memory**: Specialized storage for beauty and fashion information
- **Conscience System**: Ethical filtering and rule-based responses
- **Dream Consolidation**: Replay-based learning during "sleep" cycles

## Installation

### Prerequisites

- **Mojo programming language** (modular.com/mojo)
  - **Important**: Mojo is currently only available on macOS and Linux (including WSL)
  - Windows users: Install Windows Subsystem for Linux (WSL) to run Mojo
- Python 3.8+ (for data preprocessing)
- Git

### Setup

1. Install Mojo via Pixi (on macOS/Linux/WSL):
   ```bash
   curl -fsSL https://pixi.sh/install.sh | sh
   pixi init chatty_cathy -c https://conda.modular.com/max-nightly/ -c conda-forge
   cd chatty_cathy
   pixi add mojo
   ```

2. Clone or download the project
3. Run training: `pixi run mojo training.mojo`
4. Start interface: `pixi run mojo interface.mojo`

## Training Data

The model is designed to train on romance novel datasets:

- **Goodreads Romance Dataset**: 335K romance books + 3.5M reviews
  - Source: https://cseweb.ucsd.edu/~jmcauley/datasets/goodreads.html

- **Project Gutenberg Romance**: Public domain romance novels
  - Source: https://www.gutenberg.org/ebooks/subject/2487

- **Fashion/Beauty Focus**: Extracts passages containing fashion, makeup, and beauty keywords

## Usage

### Interactive Chat

```bash
pixi run mojo interface.mojo
```

Select option 1 for interactive mode. Commands:
- `/generate <prompt>` - Generate text
- `/fashion <item>` - Generate fashion scene
- `/romance <emotion>` - Generate romance scene
- `/emotions` - Show emotional state
- `/exit` - Quit

### API Mode

```bash
pixi run mojo interface.mojo
```

Select option 2 for API examples.

### Training

```bash
pixi run mojo training.mojo
```

Trains on romance corpus with emotional cognition guidance.

## Configuration

Training parameters in `training.mojo`:
- `vocab_size`: 50,000 (BPE tokens)
- `embed_dim`: 256-1024 (hidden dimension)
- `seq_length`: 64 (context window)
- `learning_rate`: 0.001
- `epochs`: 50+

## Model Architecture

- **Embedding Layer**: Token embeddings with fashion token boosting
- **Transformer Blocks**: Multi-head attention with emotional context
- **Emotional Overlay**: Cognition-guided generation
- **Memory Integration**: Real-time memory updates during generation

## Performance

- **Training Time**: 6-12 hours on full romance dataset
- **Memory Usage**: ~1GB for 50K vocab, scales with embed_dim
- **Generation Speed**: ~100 tokens/second (unoptimized)

## Future Improvements

- Increase embed_dim to 512-1024 for better quality
- Add more transformer layers
- Fine-tune on specific fashion/beauty corpus
- Implement model saving/loading
- Add multi-turn conversation memory
- Integrate with external fashion APIs

## Platform Notes

### Windows Users
Mojo is not currently available for native Windows. To run this project:

1. Install Windows Subsystem for Linux (WSL2)
2. Install Ubuntu or another Linux distribution from Microsoft Store
3. Follow the Linux installation instructions above
4. Run the project from within WSL

### macOS/Linux Users
Follow the standard installation instructions above.

## License

This project is for educational and research purposes. Ensure compliance with dataset licenses and ethical AI guidelines.

## Contributing

Contributions welcome! Focus areas:
- Dataset preprocessing
- Model architecture improvements
- Memory system enhancements
- Interface features

## Contact

Built with ❤️ for fashion and romance AI enthusiasts.
