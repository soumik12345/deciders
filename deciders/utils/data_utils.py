from typing import Any


def render_structured_content(value: dict[str, Any] | str | Any, indent: int = 0):
    pad = "  " * indent

    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "\n".join(
            f"{pad}- {render_structured_content(item, indent + 1).lstrip()}"
            for item in value
        )
    if isinstance(value, dict):
        return "\n".join(
            f"{pad}{key}:\n{render_structured_content(item, indent + 1)}"
            if isinstance(item, (dict, list))
            else f"{pad}{key}: {render_structured_content(item)}"
            for key, item in value.items()
        )

    raise TypeError(f"Unsupported content type: {type(value).__name__}")


def render_option_text(name, description):
    if description is None or description == "":
        return name
    return f"{name}: {render_structured_content(description)}"
