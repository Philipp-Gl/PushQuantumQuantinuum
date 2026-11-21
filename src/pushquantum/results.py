import json
from pathlib import Path
from tempfile import NamedTemporaryFile


SHOT_ARRAYS = {"results", "measurements"}


def format_json(value, depth=0, *, shot_array=False):
    indent = "  " * depth
    if isinstance(value, dict) and value and all(isinstance(key, str) for key in value):
        opening, closing = "{", "}"
        entries = []
        for key, item in value.items():
            formatted = format_json(item, depth + 1, shot_array=key in SHOT_ARRAYS)
            entries.append(f"{json.dumps(key)}: {formatted}")
    elif isinstance(value, (list, tuple)) and value:
        opening, closing = "[", "]"
        if shot_array:
            entries = [json.dumps(shot) for shot in value]
        elif any(isinstance(item, (dict, list, tuple)) for item in value):
            entries = [format_json(item, depth + 1) for item in value]
        else:
            return json.dumps(value)
    else:
        return json.dumps(value)

    nested_indent = indent + "  "
    contents = (",\n" + nested_indent).join(entries)
    return f"{opening}\n{nested_indent}{contents}\n{indent}{closing}"


def write_json(output, document):
    """Write a result document with one shot per line."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", dir=output.parent, encoding="utf-8",
                            delete_on_close=False) as temporary:
        temporary.write(format_json(document) + "\n")
        temporary.close()
        Path(temporary.name).replace(output)
