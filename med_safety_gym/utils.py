def normalize_text(text: str) -> str:
    """
    Normalize text to detect simple adversarial formatting bypasses.
    Example: "U n k n o w n" -> "Unknown"
    By only joining single character tokens, we avoid false positives 
    like joining "and we are" into "andweare".
    """
    if not text:
        return ""
    
    tokens = text.split()
    normalized_tokens = []
    temp_group = []
    
    for token in tokens:
        # Only group single alphabetical characters
        if len(token) == 1 and token.isalpha():
            temp_group.append(token)
        else:
            if temp_group:
                if len(temp_group) > 1:
                    normalized_tokens.append("".join(temp_group))
                else:
                    normalized_tokens.append(temp_group[0])
                temp_group = []
            normalized_tokens.append(token)
    
    if temp_group:
        if len(temp_group) > 1:
            normalized_tokens.append("".join(temp_group))
        else:
            normalized_tokens.append(temp_group[0])
            
    return " ".join(normalized_tokens)
