def metadata_dict_to_str(metadata: dict) -> str:
    """Convert a metadata dictionary (typically the last level of a pd.MultiIndex) to a string."""
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary.")

    # sort keys for consistency
    metadata = dict(sorted(metadata.items()))

    return " | ".join([f"{k}: {v}" for k, v in metadata.items()])

def metadata_str_to_dict(metadata_str: str) -> dict:
    """Convert a metadata string (typically the last level of a pd.MultiIndex) to a dictionary."""
    if not isinstance(metadata_str, str):
        raise ValueError("metadata_str must be a string.")
    if metadata_str == "":
        return {}
    if ":" not in metadata_str:
        raise ValueError("metadata_str must contain a ':' character.")

    metadata = {}
    for kvp in metadata_str.split(" | "):
        key, value = kvp.split(": ")
        try:
            val = float(value.strip())
        except (TypeError, ValueError):
            val = value.strip()
        metadata[key.strip()] = val

    # sort keys for consistency
    metadata = dict(sorted(metadata.items()))

    return metadata