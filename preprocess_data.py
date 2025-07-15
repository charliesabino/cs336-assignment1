#!/usr/bin/env python3
"""
Data preprocessing utility to tokenize text files and save as numpy arrays for memory-efficient training.

This script combines tokenizer training and text encoding. For more control, use the separate scripts:
- tokenize.py: Train tokenizer only
- encode.py: Encode text with existing tokenizer
"""

import argparse
import os
import pickle
import numpy as np
import subprocess
import sys


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess text data for training')
    
    parser.add_argument('--input_file', type=str, required=True, help='Input text file')
    parser.add_argument('--output_file', type=str, required=True, help='Output numpy file')
    parser.add_argument('--vocab_size', type=int, default=32000, help='Vocabulary size (default: 32000)')
    parser.add_argument('--special_tokens', nargs='*', default=['<|endoftext|>'], help='Special tokens (default: <|endoftext|>)')
    parser.add_argument('--vocab_file', type=str, default=None, help='Use existing vocabulary file instead of training')
    parser.add_argument('--merges_file', type=str, default=None, help='Use existing merges file instead of training')
    parser.add_argument('--save_tokenizer', action='store_true', help='Save trained tokenizer files')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Validate input file
    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    
    base_name = os.path.splitext(os.path.basename(args.input_file))[0]
    
    # Check if we should use existing tokenizer or train new one
    if args.vocab_file and args.merges_file:
        print("Using existing tokenizer files")
        vocab_file = args.vocab_file
        merges_file = args.merges_file
        
        # Validate tokenizer files exist
        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Vocabulary file not found: {vocab_file}")
        if not os.path.exists(merges_file):
            raise FileNotFoundError(f"Merges file not found: {merges_file}")
    else:
        print("Training new tokenizer")
        vocab_file = f"{base_name}_vocab.pkl"
        merges_file = f"{base_name}_merges.pkl"
        
        # Build tokenize command
        tokenize_cmd = [
            sys.executable, 'tokenize.py',
            '--input_file', args.input_file,
            '--vocab_size', str(args.vocab_size),
            '--vocab_output', vocab_file,
            '--merges_output', merges_file
        ]
        
        # Add special tokens
        if args.special_tokens:
            tokenize_cmd.extend(['--special_tokens'] + args.special_tokens)
        
        # Run tokenization
        print(f"Running: {' '.join(tokenize_cmd)}")
        result = subprocess.run(tokenize_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Tokenization failed: {result.stderr}")
            sys.exit(1)
        
        print(result.stdout)
    
    # Build encode command
    encode_cmd = [
        sys.executable, 'encode.py',
        '--input_file', args.input_file,
        '--output_file', args.output_file,
        '--vocab_file', vocab_file,
        '--merges_file', merges_file,
        '--add_end_token'
    ]
    
    # Add special tokens
    if args.special_tokens:
        encode_cmd.extend(['--special_tokens'] + args.special_tokens)
    
    # Run encoding
    print(f"Running: {' '.join(encode_cmd)}")
    result = subprocess.run(encode_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Encoding failed: {result.stderr}")
        sys.exit(1)
    
    print(result.stdout)
    
    # Clean up tokenizer files if not requested to save
    if not args.save_tokenizer and not (args.vocab_file and args.merges_file):
        print("Cleaning up temporary tokenizer files")
        if os.path.exists(vocab_file):
            os.remove(vocab_file)
        if os.path.exists(merges_file):
            os.remove(merges_file)
    
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()