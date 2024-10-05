# rag_system/spell_checker.py

from spellchecker import SpellChecker
import re
import sys

def initialize_spell_checker(single_word_terms):
    """
    Initialize the SpellChecker and load domain-specific single-word terms.
    """
    spell = SpellChecker()
    spell.word_frequency.load_words(single_word_terms)
    return spell

def protect_domain_terms(text, domain_terms):
    """
    Replace domain-specific multi-word terms with placeholders.
    """
    placeholders = {}
    for idx, term in enumerate(domain_terms):
        placeholder = f"__TERM_{idx}__"
        placeholders[placeholder] = term
        # Use word boundaries to avoid partial replacements
        pattern = r'\b' + re.escape(term) + r'\b'
        text = re.sub(pattern, placeholder, text)
    return text, placeholders

def restore_domain_terms(text, placeholders):
    """
    Replace placeholders with the original domain-specific terms.
    """
    for placeholder, term in placeholders.items():
        text = text.replace(placeholder, term)
    return text

def correct_text_spellchecker(text, domain_terms, spell):
    """
    Correct spelling mistakes in the text using pyspellchecker,
    while preserving domain-specific terms.
    """
    # Protect multi-word domain-specific terms
    protected_text, placeholders = protect_domain_terms(text, domain_terms)
    
    # Split text into words
    words = protected_text.split()
    
    corrected_words = []
    for word in words:
        # Skip placeholders
        if word in placeholders:
            corrected_words.append(word)
            continue
        
        # Check if the word is capitalized (proper noun) and in the spell checker's dictionary
        if word.istitle() and word in spell:
            corrected_words.append(word)
            continue
        
        # Correct the word if it's misspelled
        if word.lower() in spell:
            corrected_words.append(word)
        else:
            corrected = spell.correction(word)
            corrected_words.append(corrected if corrected else word)
    
    # Reconstruct the text
    corrected_text = ' '.join(corrected_words)
    
    # Restore the original domain-specific terms
    corrected_text = restore_domain_terms(corrected_text, placeholders)
    
    return corrected_text