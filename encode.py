#!/usr/bin/env python3
"""
Standalone encoding script for converting text to tokens using existing tokenizer.
Separated from preprocessing to allow independent text encoding.
"""

import argparse
import os
import pickle
import numpy as np
from cs336_basics.tokenizer import Tokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Encode text using existing BPE tokenizer')
    
    parser.add_argument('--input_file', type=str, required=True, help='Input text file')
    parser.add_argument('--output_file', type=str, required=True, help='Output numpy file')
    parser.add_argument('--vocab_file', type=str, required=True, help='Vocabulary file (.pkl)')
    parser.add_argument('--merges_file', type=str, required=True, help='Merges file (.pkl)')
    parser.add_argument('--special_tokens', nargs='*', default=['<|endoftext|>'], help='Special tokens (default: <|endoftext|>)')
    parser.add_argument('--add_end_token', action='store_true', help='Add end of text token to the end')
    parser.add_argument('--end_of_text', type=str, default='<|endoftext|>', help='End of text marker (default: <|endoftext|>)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate input files
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    if not os.path.exists(args.vocab_file):
        raise FileNotFoundError(f"Vocabulary file not found: {args.vocab_file}")
    if not os.path.exists(args.merges_file):
        raise FileNotFoundError(f"Merges file not found: {args.merges_file}")
    
    # Load tokenizer
    print(f"Loading vocabulary from {args.vocab_file}")
    with open(args.vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    
    print(f"Loading merges from {args.merges_file}")
    with open(args.merges_file, 'rb') as f:
        merges = pickle.load(f)
    
    # Create tokenizer
    tokenizer = Tokenizer(vocab, merges, args.special_tokens)
    
    # Read text
    print(f"Reading text from {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Optionally add end of text token
    if args.add_end_token:
        text += args.end_of_text
    
    # Encode text
    print(f"Encoding text... (length: {len(text)} characters)")
    tokens = tokenizer.encode(text)
    
    print(f"Encoded to {len(tokens)} tokens")
    
    # Convert to numpy array and save
    tokens_array = np.array(tokens, dtype=np.int32)
    
    print(f"Saving encoded data to {args.output_file}")
    np.save(args.output_file, tokens_array)
    
    print(f"Encoding complete!")
    print(f"Output shape: {tokens_array.shape}")
    print(f"Output dtype: {tokens_array.dtype}")


if __name__ == "__main__":
    main()