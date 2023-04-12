from pathlib import Path

def get_path():
    path = {}
    with open(Path(__file__).parent / "path.sh", "r") as f:
        for line in f.readlines():
            if line != "\n":
                name, value = line.split("=")
                value = value.rstrip("\n")
                for k,v in path.items():
                    value = value.replace(f"${k}", str(v))
                path[name] = Path(value)
    return path
