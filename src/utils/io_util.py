import yaml


def load_yaml(file_path, verbose=True):
    with open(file_path, "r") as f:
        yml_file = yaml.load(f, Loader=yaml.SafeLoader)
    if verbose:
        print("Load yaml file from {}".format(file_path))
    return yml_file
