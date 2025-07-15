#!/usr/bin/env python3
"""
Data preprocessing utility to tokenize text files and save as numpy arrays for memory-efficient training.
"""

import argparse
import os
import pickle
import numpy as np
from cs336_basics.tokenizer import Tokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess text data for training')
    
    parser.add_argument('--input_file', type=str, required=True, help='Input text file')
    parser.add_argument('--output_file', type=str, required=True, help='Output numpy file')
    parser.add_argument('--vocab_file', type=str, required=True, help='Vocabulary file (.pkl)')
    parser.add_argument('--merges_file', type=str, required=True, help='Merges file (.pkl)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load tokenizer
    print(f"Loading vocabulary from {args.vocab_file}")
    with open(args.vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    
    print(f"Loading merges from {args.merges_file}")
    with open(args.merges_file, 'rb') as f:
        merges = pickle.load(f)
    
    tokenizer = Tokenizer(vocab, merges)
    
    # Read and tokenize text
    print(f"Reading text from {args.input_file}")
    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Tokenizing text... (length: {len(text)} characters)")
    tokens = tokenizer.encode(text)
    
    print(f"Tokenized to {len(tokens)} tokens")
    
    # Convert to numpy array and save
    tokens_array = np.array(tokens, dtype=np.int32)
    
    print(f"Saving tokenized data to {args.output_file}")
    np.save(args.output_file, tokens_array)
    
    print(f"Preprocessing complete!")
    print(f"Output shape: {tokens_array.shape}")
    print(f"Output dtype: {tokens_array.dtype}")


if __name__ == "__main__":
    main()