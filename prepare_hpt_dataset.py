"""
HPT Dataset Preparation Script
Extracts individual word images from scanned pages using bounding box coordinates
"""

import os
import pandas as pd
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm

def parse_word_places(word_places_path):
    """Parse the word_places.txt file and return a DataFrame"""
    data = []
    
    with open(word_places_path, 'r', encoding='latin-1') as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if line.startswith('%') or not line:
                continue
            
            # Parse the line: "filename" word top_line top_col bottom_line bottom_col
            parts = line.split()
            if len(parts) < 6:
                continue
            
            # Extract filename (remove quotes)
            filename = parts[0].strip('"')
            word = parts[1]
            
            # Extract coordinates
            top_line = int(parts[2])
            top_col = int(parts[3])
            bottom_line = int(parts[4])
            bottom_col = int(parts[5])
            
            data.append({
                'filename': filename,
                'word': word,
                'top_line': top_line,
                'top_col': top_col,
                'bottom_line': bottom_line,
                'bottom_col': bottom_col
            })
    
    return pd.DataFrame(data)

def extract_word_images(base_path, word_places_df, output_dir, author_name):
    """Extract individual word images from scanned pages"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    dataset_records = []
    
    for idx, row in tqdm(word_places_df.iterrows(), total=len(word_places_df), 
                         desc=f"Processing {author_name}"):
        try:
            # Construct the full image path
            image_path = os.path.join(base_path, row['filename'].replace('\\', os.sep))
            
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            
            # Load the image
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            
            # Extract the word region using bounding box
            # Note: PIL uses (left, upper, right, lower) format
            word_img = img.crop((
                row['top_col'],      # left
                row['top_line'],     # upper
                row['bottom_col'],   # right
                row['bottom_line']   # lower
            ))
            
            # Create output filename
            word_clean = row['word'].replace('\\', '_').replace('/', '_')
            output_filename = f"{author_name}_word_{idx:05d}_{word_clean}.png"
            output_filepath = output_path / output_filename
            
            # Save the extracted word image
            word_img.save(output_filepath)
            
            # Record the dataset entry
            dataset_records.append({
                'path': str(output_filepath),
                'word': row['word'],
                'author': author_name,
                'original_image': row['filename'],
                'width': word_img.width,
                'height': word_img.height
            })
            
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue
    
    return pd.DataFrame(dataset_records)

def main():
    """Main function to prepare the dataset"""
    
    # Configuration
    base_dataset_path = "HPT_handwritten_polish_text_dataset"
    output_dir = "hpt_extracted_words"
    
    # List of authors
    authors = [f"author{i}" for i in range(1, 9)]
    
    all_dataset_records = []
    
    for author in authors:
        print(f"\n{'='*60}")
        print(f"Processing {author}")
        print(f"{'='*60}")
        
        author_path = os.path.join(base_dataset_path, author)
        word_places_file = os.path.join(author_path, "word_places.txt")
        
        if not os.path.exists(word_places_file):
            print(f"Warning: word_places.txt not found for {author}")
            continue
        
        # Parse word places
        word_places_df = parse_word_places(word_places_file)
        print(f"Found {len(word_places_df)} words for {author}")
        
        # Extract word images
        author_output_dir = os.path.join(output_dir, author)
        dataset_df = extract_word_images(author_path, word_places_df, 
                                        author_output_dir, author)
        
        all_dataset_records.append(dataset_df)
        print(f"Successfully extracted {len(dataset_df)} word images for {author}")
    
    # Combine all records
    final_dataset = pd.concat(all_dataset_records, ignore_index=True)
    
    # Save the dataset CSV
    output_csv = "hpt_dataset.csv"
    final_dataset.to_csv(output_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"Dataset preparation complete!")
    print(f"{'='*60}")
    print(f"Total words extracted: {len(final_dataset)}")
    print(f"Unique words: {final_dataset['word'].nunique()}")
    print(f"Dataset saved to: {output_csv}")
    
    # Print statistics
    print(f"\nWord frequency distribution:")
    word_counts = final_dataset['word'].value_counts()
    print(f"Words appearing once: {(word_counts == 1).sum()}")
    print(f"Words appearing 2+ times: {(word_counts >= 2).sum()}")
    print(f"Words appearing 3+ times: {(word_counts >= 3).sum()}")
    
    print(f"\nTop 20 most frequent words:")
    print(word_counts.head(20))
    
    print(f"\nAuthor distribution:")
    print(final_dataset['author'].value_counts())

if __name__ == "__main__":
    main()