import os


def remap_keys(mapping):
    return [{'key': k, 'value': v} for k, v in mapping.items()]


def all_json_files():
    directory = os.fsencode(str("data"))
    return [os.path.join(directory, file) for file in os.listdir(
        directory) if os.fsdecode(file).endswith(".json")]


# print(all_json_files())
