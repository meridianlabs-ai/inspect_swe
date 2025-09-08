from datetime import date, datetime, time
from typing import Any, Dict, List


def to_toml(data: Dict[str, Any]) -> str:
    """Convert a dictionary to TOML format string."""
    lines = []

    # Handle top-level key-value pairs first
    top_level = {}
    tables = {}

    for key, value in data.items():
        if isinstance(value, dict):
            tables[key] = value
        else:
            top_level[key] = value

    # Write top-level pairs
    for key, value in top_level.items():
        lines.append(f"{key} = {_format_value(value)}")

    # Write tables
    for table_name, table_data in tables.items():
        if lines:  # Add blank line before tables
            lines.append("")
        lines.append(f"[{table_name}]")
        _write_table(lines, table_data)

    return "\n".join(lines)


def _format_value(value: Any) -> str:
    """Format a Python value as TOML."""
    if isinstance(value, str):
        # Escape special characters and quote
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, datetime):
        return value.isoformat()
    elif isinstance(value, date):
        return value.isoformat()
    elif isinstance(value, time):
        return value.isoformat()
    elif isinstance(value, list):
        formatted_items = [_format_value(item) for item in value]
        return f"[{', '.join(formatted_items)}]"
    elif value is None:
        raise ValueError("TOML doesn't support null values")
    else:
        raise TypeError(f"Unsupported type: {type(value)}")


def _write_table(lines: List[str], data: Dict[str, Any]) -> None:
    """Write table contents."""
    for key, value in data.items():
        if not isinstance(value, dict):
            lines.append(f"{key} = {_format_value(value)}")
