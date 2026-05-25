#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
normalizer.py
-------------
A script to clean, format, and normalize Vietnamese text (e.g., converting
numbers to words, resolving abbreviations, removing unsupported symbols)
to ensure smooth phonetic mapping for TTS training.
"""

import re
import argparse
from num2words import num2words

# Basic Vietnamese abbreviation expansion mapping
ABBREVIATIONS = {
    r"\bkm\b": "ki lô mét",
    r"\bm\b": "mét",
    r"\bcm\b": "xen ti mét",
    r"\bmm\b": "mi li mét",
    r"\bkg\b": "ki lô gam",
    r"\bg\b": "gam",
    r"\bđ\b": "đồng",
    r"\bđđ\b": "địa điểm",
    r"\bsđt\b": "số điện thoại",
    r"\btp\b": "thành phố",
    r"\bhn\b": "hà nội",
    r"\bhcm\b": "hồ chí minh",
    r"\bđh\b": "đại học",
    r"\bths\b": "thạc sĩ",
    r"\bts\b": "tiến sĩ",
    r"\bgs\b": "giáo sư",
    r"\bpgs\b": "phó giáo sư",
    r"\bnsnd\b": "nghệ sĩ nhân dân",
    r"\bnsưt\b": "nghệ sĩ ưu tú",
    r"\bvs\b": "với",
    r"\b&\b": "và",
    r"\b%\b": "phần trăm",
    r"\bco\b": "công ty",
}

def replace_abbreviations(text):
    """
    Expands abbreviations in Vietnamese text.
    """
    for pattern, expansion in ABBREVIATIONS.items():
        text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
    return text

def convert_number_match(match):
    """
    Callback function to convert a matched number string into Vietnamese words.
    """
    number_str = match.group(0)
    try:
        # Convert floating point numbers
        if '.' in number_str or ',' in number_str:
            # Replace comma separators with dots for float parser if needed
            cleaned_num = number_str.replace(',', '.')
            val = float(cleaned_num)
            return num2words(val, lang='vi')
        else:
            val = int(number_str)
            return num2words(val, lang='vi')
    except Exception:
        # Fallback to returning original string if conversion fails
        return number_str

def normalize_numbers(text):
    """
    Locates numeric values in the text and converts them into spoken Vietnamese words.
    Matches integers and simple floating point numbers (e.g. 123, 4.5, 4,5).
    """
    # Regex matching integers and floats
    number_pattern = re.compile(r'\b\d+(?:[\.,]\d+)?\b')
    return number_pattern.sub(convert_number_match, text)

def clean_special_characters(text):
    """
    Removes unsupported punctuation and symbols from text, leaving only words and standard punctuation.
    """
    # Allow alphanumeric characters, Vietnamese accents, spaces, and basic punctuation
    # Vietnamese range: a-z, A-Z, 0-9, spaces, and Vietnamese accented characters:
    # àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđĐ
    allowed_chars = re.compile(r'[^a-zA-Z0-9aAàÀảẢãÃạẠăĂằẰắẮẳẲẵẴặẶâÂầẦấẤẩẨẫẪậẬèÈéÉẻẺẽẼẹẸêÊềỀếẾểỂễỄệỆìÌíÍỉỈĩĨịỊòÒóÓỏỎõÕọỌôÔồỒốỐổỔỗỖộỘơƠờỜớỚởỞỡỠợỢùÙúÚủỦũŨụỤưƯừỪứỨửỬữỮựỰỳỲýÝỷỶỹỸỵYđĐ\s\.,\?!\-\'\"]')
    text = allowed_chars.sub(' ', text)
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_text(text):
    """
    Runs the complete normalization pipeline for a line of text.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase text
    text = text.lower()
    
    # 2. Expand abbreviations
    text = replace_abbreviations(text)
    
    # 3. Convert numbers to words
    text = normalize_numbers(text)
    
    # 4. Filter special characters
    text = clean_special_characters(text)
    
    return text

def main():
    parser = argparse.ArgumentParser(description="Normalize Vietnamese text by expanding abbreviations and converting numbers to words.")
    parser.add_argument("--text", type=str, help="Text string to normalize.")
    parser.add_argument("--input_file", type=str, help="Input text file to normalize line-by-line.")
    parser.add_argument("--output_file", type=str, help="Output file to write normalized text.")
    
    args = parser.parse_args()
    
    if args.text:
        print(normalize_text(args.text))
    elif args.input_file:
        if not os.path.exists(args.input_file):
            print(f"[-] Input file not found: {args.input_file}")
            return
            
        with open(args.input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        normalized_lines = [normalize_text(line) for line in lines]
        
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                for line in normalized_lines:
                    f.write(line + "\n")
            print(f"[+] Normalized text written to: {args.output_file}")
        else:
            for line in normalized_lines:
                print(line)
    else:
        # Interactive mode helper
        print("[*] Normalizer loaded. Type text to test (Ctrl+C to exit):")
        try:
            while True:
                user_input = input("> ")
                print(normalize_text(user_input))
        except KeyboardInterrupt:
            print("\nExiting.")

if __name__ == "__main__":
    main()
