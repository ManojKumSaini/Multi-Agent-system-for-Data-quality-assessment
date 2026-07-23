import pandas as pd
import numpy as np
import re
import os
import sys
import html

# Ensure output directory exists
os.makedirs('outputs/preprocessing/', exist_ok=True)

# Load data
try:
    df = pd.read_excel('data/NIFTY.xlsx')
except Exception as e:
    print(f"STATUS: ERROR � {str(e)}")
    sys.exit(1)

loaded = len(df)

# Check required columns
required_cols = ['id', 'date', 'news', 'label', 'pct_change']
if not all(col in df.columns for col in required_cols):
    print("STATUS: ERROR � Missing required columns")
    sys.exit(1)

# Create preprocessed_text column
df['preprocessed_text'] = df['news'].copy()

# Mandatory operations 1-7
# 1. Handle nulls
df['preprocessed_text'] = df['preprocessed_text'].fillna("").astype(str).str.strip()

# 2. Remove URLs
df['preprocessed_text'] = df['preprocessed_text'].str.replace(r'https?://\S+|www\.\S+', ' ', regex=True)

# 3. Remove emails
df['preprocessed_text'] = df['preprocessed_text'].str.replace(r'\S+@\S+', ' ', regex=True)

# 4. Remove HTML tags
df['preprocessed_text'] = df['preprocessed_text'].str.replace(r'<[^>]*>', ' ', regex=True)

# 5. Fix encoding artifacts
df['preprocessed_text'] = df['preprocessed_text'].str.replace(r'’|é|�|\\x00', '', regex=True)

# 6. Unescape HTML entities
df['preprocessed_text'] = df['preprocessed_text'].apply(html.unescape)

# 7. Lowercase
df['preprocessed_text'] = df['preprocessed_text'].str.lower()

# Optional operations
# Check for possessives
if df['preprocessed_text'].str.contains(r"['\u2019]s\b", regex=True, na=False).any():
    df['preprocessed_text'] = df['preprocessed_text'].str.replace(r"['\u2019]s\b", ' ', regex=True)
    # Justified: dataset contains possessive forms like 'company's'

# Check for plural indicators
if df['preprocessed_text'].str.contains(r'\(s\)', regex=True, na=False).any():
    df['preprocessed_text'] = df['preprocessed_text'].str.replace(r'\(s\)', ' ', regex=True)
    # Justified: dataset contains constructions like 'item(s)'

# Check for remaining apostrophes
if df['preprocessed_text'].str.contains(r"['\u2019]", regex=True, na=False).any():
    df['preprocessed_text'] = df['preprocessed_text'].str.replace(r"['\u2019]", ' ', regex=True)
    # Justified: remaining apostrophes add noise

# Check for ellipses
if df['preprocessed_text'].str.contains(r'\.{3,}|\u2026', regex=True, na=False).any():
    df['preprocessed_text'] = df['preprocessed_text'].str.replace(r'\.{3,}|\u2026', ' ', regex=True)
    # Justified: dataset contains ellipsis characters

# Check for special characters
if df['preprocessed_text'].str.contains(r'[^a-z0-9\s\.\$&\u20ac%,-]', regex=True, na=False).any():
    df['preprocessed_text'] = df['preprocessed_text'].str.replace(r'[^a-z0-9\s\.\$&\u20ac%,-]', ' ', regex=True)
    # Justified: dataset contains special characters not meaningful for financial domain

# Check for gibberish
if df['preprocessed_text'].str.contains(r'\b\w{25,}\b', regex=True, na=False).any():
    df['preprocessed_text'] = df['preprocessed_text'].str.replace(r'\b\w{25,}\b', ' ', regex=True)
    # Justified: dataset contains corrupted tokens

# Mandatory operation 8: Collapse whitespace and strip
df['preprocessed_text'] = df['preprocessed_text'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Post-cleaning filters
# 1. Remove empty rows
df = df[df['preprocessed_text'] != ""]

# 2. Minimum word count filter (threshold=3 for financial news)
df['word_count'] = df['preprocessed_text'].str.split().str.len()
df = df[df['word_count'] >= 3]
after_min_word_filter = len(df)
del df['word_count']

# 3. Deduplication
df = df.drop_duplicates(subset=['preprocessed_text', 'date'])
after_dedup = len(df)

# Check for boilerplate
boilerplate_rows = df.groupby('preprocessed_text').filter(lambda x: len(x) > 50)
if not boilerplate_rows.empty:
    df['is_boilerplate'] = df['preprocessed_text'].isin(boilerplate_rows['preprocessed_text'])
    after_empty_removal = len(df)
else:
    after_empty_removal = len(df)
    after_min_word_filter = len(df)
    after_dedup = len(df)

# Final column selection
columns = ['id', 'date', 'news', 'label', 'pct_change', 'preprocessed_text']
if 'is_boilerplate' in df.columns:
    columns.append('is_boilerplate')

# Cast to string and save
df['preprocessed_text'] = df['preprocessed_text'].astype(str)
df.to_csv('outputs/preprocessing/NIFTY_preprocessed.csv', columns=columns, index=False)

# Print statistics
print(f"loaded: {loaded}")
print(f"after_empty_removal: {after_empty_removal}")
print(f"after_min_word_filter: {after_min_word_filter}")
print(f"after_dedup: {after_dedup}")
print("STATUS: Phase 1 Script Completed.")