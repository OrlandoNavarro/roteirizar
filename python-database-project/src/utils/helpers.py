def validate_data(data):
    if not isinstance(data, dict):
        raise ValueError("Data must be a dictionary.")
    
    if 'name' not in data or 'value' not in data:
        raise ValueError("Data must contain 'name' and 'value' keys.")
    
    return True

def format_data(data):
    return {key: str(value).strip() for key, value in data.items()}