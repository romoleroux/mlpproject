import os

IGNORE = {"venv", "__pycache__", ".git", ".ipynb_checkpoints"}

def print_tree(start_path, prefix=""):
    items = [i for i in sorted(os.listdir(start_path)) if i not in IGNORE]
    for i, item in enumerate(items):
        path = os.path.join(start_path, item)
        connector = "\\-- " if i == len(items) - 1 else "|-- "
        print(prefix + connector + item)
        if os.path.isdir(path):
            extension = "    " if i == len(items) - 1 else "|   "
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    print_tree(".")
