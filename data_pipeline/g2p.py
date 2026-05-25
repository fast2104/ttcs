#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
g2p.py
------
Vietnamese Grapheme-to-Phoneme (G2P) converter. Converts normalized text
into phonemic sequences (IPA representation or simplified tokens) to assist
TTS models in learning correct pronunciation and tone alignment.
"""

import re
import sys

# Define vowels and corresponding tone markings
# Tones: 0: Ngang (none), 1: Huyền, 2: Sắc, 3: Hỏi, 4: Ngã, 5: Nặng
TONE_MAP = {
    # Unmarked (Ngang) - Tone 0
    # Huyền - Tone 1
    'à': ('a', 1), 'ầ': ('â', 1), 'ằ': ('ă', 1), 'è': ('e', 1), 'ề': ('ê', 1),
    'ì': ('i', 1), 'ò': ('o', 1), 'ồ': ('ô', 1), 'ờ': ('ơ', 1), 'ù': ('u', 1),
    'ừ': ('ư', 1), 'ỳ': ('y', 1),
    # Sắc - Tone 2
    'á': ('a', 2), 'ấ': ('â', 2), 'ắ': ('ă', 2), 'é': ('e', 2), 'ế': ('ê', 2),
    'í': ('i', 2), 'ó': ('o', 2), 'ố': ('ô', 2), 'ớ': ('ơ', 2), 'ú': ('u', 2),
    'ứ': ('ư', 2), 'ý': ('y', 2),
    # Hỏi - Tone 3
    'ả': ('a', 3), 'ẩ': ('â', 3), 'ẳ': ('ă', 3), 'ẻ': ('e', 3), 'ể': ('ê', 3),
    'ỉ': ('i', 3), 'ỏ': ('o', 3), 'ổ': ('ô', 3), 'ở': ('ơ', 3), 'ủ': ('u', 3),
    'ử': ('ư', 3), 'ỷ': ('y', 3),
    # Ngã - Tone 4
    'ã': ('a', 4), 'ẫ': ('â', 4), 'ẵ': ('ă', 4), 'ẽ': ('e', 4), 'ễ': ('ê', 4),
    'ĩ': ('i', 4), 'õ': ('o', 4), 'ỗ': ('ô', 4), 'ỡ': ('ơ', 4), 'ũ': ('u', 4),
    'ữ': ('ư', 4), 'ỹ': ('y', 4),
    # Nặng - Tone 5
    'ạ': ('a', 5), 'ậ': ('â', 5), 'ặ': ('ă', 5), 'ẹ': ('e', 5), 'ệ': ('ê', 5),
    'ị': ('i', 5), 'ọ': ('o', 5), 'ộ': ('ô', 5), 'ợ': ('ơ', 5), 'ụ': ('u', 5),
    'ự': ('ư', 5), 'ỵ': ('y', 5),
}

# Consonants and Vowels maps
INITIAL_CONSONANTS = [
    'ch', 'gh', 'kh', 'ngh', 'ng', 'nh', 'ph', 'th', 'tr', 'gi', 'qu',
    'b', 'c', 'd', 'đ', 'g', 'h', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'v', 'x'
]

CODA_CONSONANTS = ['ch', 'nh', 'ng', 'c', 'm', 'n', 'p', 't', 'o', 'u', 'y']

def extract_tone_and_clean_vowel(char):
    """
    Checks if a character has a tone. Returns the clean vowel and tone index.
    """
    if char in TONE_MAP:
        return TONE_MAP[char]
    return char, 0

def split_syllable(word):
    """
    Splits a single lowercase Vietnamese syllable/word into:
    (Initial Consonant, Nucleus Vowels, Coda Consonants, Tone)
    """
    word = word.lower().strip()
    
    # 1. Match Initial Consonants
    initial = ""
    for c in INITIAL_CONSONANTS:
        if word.startswith(c):
            # Check edge cases (e.g. 'qu' vs 'q')
            initial = c
            word = word[len(c):]
            break
            
    # 2. Extract tone and clean vowels from the remaining characters
    tone = 0
    cleaned_vowels_list = []
    
    # We examine each remaining character.
    # If it is a vowel, we extract tone.
    for char in word:
        clean_char, char_tone = extract_tone_and_clean_vowel(char)
        if char_tone > 0:
            tone = char_tone
        cleaned_vowels_list.append(clean_char)
        
    cleaned_remain = "".join(cleaned_vowels_list)
    
    # 3. Identify Vowel Nucleus vs Coda Consonants
    # Vowels: a, ă, â, e, ê, i, o, ô, ơ, u, ư, y
    vowel_regex = re.compile(r'[aăâeêioôơuưy]+')
    vowel_match = vowel_regex.search(cleaned_remain)
    
    if not vowel_match:
        # Fallback if no vowels found
        return initial, cleaned_remain, "", tone
        
    nucleus = vowel_match.group(0)
    coda = cleaned_remain[vowel_match.end():]
    
    # Special adjustment: if nucleus is 'gi', the 'g' might be initial and 'i' nucleus
    if initial == 'gi' and not nucleus:
        initial = 'g'
        nucleus = 'i'
        
    return initial, nucleus, coda, tone

def phonemize_word(word):
    """
    Converts a single Vietnamese word into its phoneme representation.
    """
    # Exclude punctuation
    if not re.match(r'^[a-zA-Z0-9aAàÀảẢãÃạẠăĂằẰắẮẳẲẵẴặẶâÂầẦấẤẩẨẫẪậẬèÈéÉẻẺẽẼẹẸêÊềỀếẾểỂễỄệỆìÌíÍỉỈĩĨịỊòÒóÓỏỎõÕọỌôÔồỒốỐổỔỗỖộỘơƠờỜớỚởỞỡỠợỢùÙúÚủỦũŨụỤưƯừỪứỨửỬữỮựỰỳỲýÝỷỶỹỸỵYđĐ]+$', word):
        return [word]
        
    initial, nucleus, coda, tone = split_syllable(word)
    
    # Format phonemes: [initial, nucleus, coda, f"t{tone}"]
    # Only keep non-empty entries
    phonemes = []
    if initial:
        phonemes.append(initial)
    if nucleus:
        phonemes.append(nucleus)
    if coda:
        phonemes.append(coda)
    
    # Append tone label (e.g. t0, t1, t2)
    phonemes.append(f"t{tone}")
    
    return phonemes

def text_to_phonemes(text):
    """
    Converts a sentence to a sequence of phonemes.
    """
    words = text.split()
    output_phonemes = []
    for word in words:
        # Remove trailing/leading punctuation
        cleaned_word = re.sub(r'^\W+|\W+$', '', word)
        p = phonemize_word(cleaned_word)
        output_phonemes.extend(p)
        # Add a word-boundary token
        output_phonemes.append("sp") # short pause / space boundary
        
    # Remove trailing word boundary token
    if output_phonemes and output_phonemes[-1] == "sp":
        output_phonemes.pop()
        
    return output_phonemes

def main():
    parser = argparse.ArgumentParser(description="Convert normalized Vietnamese text to phoneme sequences.")
    parser.add_argument("--text", type=str, help="Text to convert.")
    
    args = parser.parse_args()
    
    # Try importing viphoneme to see if it is compiled/accessible
    try:
        import viphoneme
        has_viphoneme = True
    except ImportError:
        has_viphoneme = False
        
    if args.text:
        if has_viphoneme:
            # viphoneme is available, use it (produces IPA)
            print("[*] Using viphoneme (IPA):")
            print(viphoneme.vi2ipa(args.text))
        else:
            # Fallback to rules-based converter
            print("[*] Using standalone rules-based phonemizer:")
            ph_seq = text_to_phonemes(args.text)
            print(" ".join(ph_seq))
    else:
        print("[*] G2P Phonemizer. Type sentence to test (Ctrl+C to exit):")
        try:
            while True:
                user_input = input("> ")
                if has_viphoneme:
                    import viphoneme
                    print("IPA:", viphoneme.vi2ipa(user_input))
                else:
                    ph_seq = text_to_phonemes(user_input)
                    print("Phonemes:", "/".join(ph_seq))
        except KeyboardInterrupt:
            print("\nExiting.")

if __name__ == "__main__":
    main()
