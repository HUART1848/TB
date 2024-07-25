import itertools

def params_product(**params):
    keys = params.keys()
    for instance in itertools.product(*params.values()):
        yield dict(zip(keys, instance))

def append_to_file(filename: str, content: str):
    with open(filename, "a") as f:
        f.write(content)
