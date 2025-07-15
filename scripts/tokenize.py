#!/usr/bin/env python3
"""
Standalone tokenization script for training BPE tokenizers.
Separated from preprocessing to allow independent tokenizer training.
"""

import argparse
import os
import pickle
from cs336_basics.tokenizer import tokenize


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train BPE tokenizer from text data')
    
    parser.add_argument('--input_file', type=str, required=True, help='Input text file')
    parser.add_argument('--vocab_size', type=int, default=32000, help='Vocabulary size (default: 32000)')
    parser.add_argument('--special_tokens', nargs='*', default=['<|endoftext|>'], help='Special tokens (default: <|endoftext|>)')
    parser.add_argument('--vocab_output', type=str, default=None, help='Output vocabulary file (default: auto-generated)')
    parser.add_argument('--merges_output', type=str, default=None, help='Output merges file (default: auto-generated)')
    parser.add_argument('--end_of_text', type=str, default='<|endoftext|>', help='End of text marker (default: <|endoftext|>)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    # Auto-generate output filenames if not provided
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    vocab_output = args.vocab_output or f"{base_name}_vocab.pkl"
    merges_output = args.merges_output or f"{base_name}_merges.pkl"
    
    # Train tokenizer
    print(f"Training tokenizer from {args.input_file}")
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Special tokens: {args.special_tokens}")
    print(f"End of text marker: {args.end_of_text}")
    
    vocab, merges = tokenize(
        args.input_file,
        args.vocab_size,
        args.special_tokens
    )
    
    # Save vocabulary
    print(f"Saving vocabulary to {vocab_output}")
    with open(vocab_output, 'wb') as f:
        pickle.dump(vocab, f)
    
    # Save merges
    print(f"Saving merges to {merges_output}")
    with open(merges_output, 'wb') as f:
        pickle.dump(merges, f)
    
    print(f"Tokenizer training complete!")
    print(f"Final vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print(f"Files created:")
    print(f"  - {vocab_output}")
    print(f"  - {merges_output}")


if __name__ == "__main__":
    main()