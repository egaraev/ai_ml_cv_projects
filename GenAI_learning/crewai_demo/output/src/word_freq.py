def count_words(text: str) -> dict[str, int]:
    """
    Count the frequency of each word in the given text.
    
    This function processes the input text by:
    1. Converting to lowercase
    2. Replacing all non-alphabetic characters with spaces
    3. Splitting on whitespace
    4. Counting non-empty tokens
    
    Args:
        text: Input string to analyze
        
    Returns:
        Dictionary mapping words to their frequencies
        
    Examples:
        >>> count_words("Hello, world!")
        {'hello': 1, 'world': 1}
        >>> count_words("Hello, world! Hello.")
        {'hello': 2, 'world': 1}
    """
    # Handle edge cases
    if text is None or text == "":
        return {}
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace non-alphabetic characters with spaces (keeping hyphens)
    cleaned_text = ""
    for char in text:
        if char.isalpha() or char == '-':
            cleaned_text += char
        else:
            cleaned_text += " "
    
    # Split on whitespace and count non-empty tokens
    words = cleaned_text.split()
    word_count = {}
    
    for word in words:
        if word:  # Ensure word is not empty
            # Handle hyphenated words by splitting them
            if '-' in word:
                parts = word.split('-')
                for part in parts:
                    if part:  # Only count non-empty parts
                        word_count[part] = word_count.get(part, 0) + 1
            else:
                word_count[word] = word_count.get(word, 0) + 1
    
    return word_count