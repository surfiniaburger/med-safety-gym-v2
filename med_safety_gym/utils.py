def normalize_text(text: str) -> str:
    """
    Normalize text to detect adversarial formatting bypasses.
    Example: "U n k n o w n" -> "Unknown"
    Example: "Un kn own d r ug" -> "Unknowndrug"
    """
    if not text:
        return ""
    
    tokens = text.split()
    normalized_tokens = []
    temp_group = []
    
    for token in tokens:
        # Fragments are typically 1-3 characters
        if len(token) <= 3:
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
