import re

import yaml


def read_front_matter_name(content: str) -> str | None:
    # front-matter
    frontmatter_match = re.match(r"^\s*---\s*\n(.*?)\n---", content, re.DOTALL)
    if not frontmatter_match:
        return None
    frontmatter = frontmatter_match.group(1)

    try:
        # Parse as YAML
        data = yaml.safe_load(frontmatter)
        if "name" in data:
            return str(data.get("name"))
        else:
            return None
    except yaml.YAMLError:
        return None
