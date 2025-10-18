#!/usr/bin/env python3
"""
chatty_cathy Python Implementation
Fashion-focused romance language model with emotional cognition

This is the Python equivalent of the Mojo version, using PyTorch.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training import main as train_main

if __name__ == "__main__":
    print("🚀 Starting chatty_cathy Python Training")
    print("=" * 50)

    try:
        train_main()
        print("\n✅ Training completed successfully!")
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        sys.exit(1)