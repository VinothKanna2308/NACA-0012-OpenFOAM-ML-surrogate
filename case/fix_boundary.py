#!/usr/bin/env python3
"""Set patch types in constant/polyMesh/boundary (airfoil -> wall, frontAndBack -> empty).

gmshToFoam creates every patch as type 'patch'.  This does the job of changeDictionary
without depending on its behaviour, which differs between OpenFOAM releases.

    python3 fix_boundary.py <case_dir>
"""
import re
import sys

TYPES = {"airfoil": "wall", "frontAndBack": "empty"}


def fix(text):
    for name, ptype in TYPES.items():
        pat = re.compile(r"(\b" + re.escape(name) + r"\s*\{)(.*?)(\})", re.S)
        m = pat.search(text)
        if m is None:
            raise SystemExit(f"patch {name} not found in boundary file")
        body = m.group(2)
        body = re.sub(r"\n\s*type\s+\w+;", "", body)
        body = re.sub(r"\n\s*inGroups\s+[^;]*;", "", body)
        body = re.sub(r"\n\s*physicalType\s+\w+;", "", body)
        body = f"\n        type            {ptype};\n        inGroups        List<word> 1({ptype});" + body
        text = text[: m.start()] + m.group(1) + body + m.group(3) + text[m.end():]
    return text


if __name__ == "__main__":
    path = sys.argv[1].rstrip("/") + "/constant/polyMesh/boundary"
    with open(path) as f:
        text = f.read()
    with open(path, "w") as f:
        f.write(fix(text))
    print("patched", path)
