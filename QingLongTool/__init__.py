import configparser
import ast
from . import nbody
from . import lensing
from . import NG 

def read_config(filepath):
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(filepath)

    params = {}
    section = 'cosmology_model'
    for option in config.options(section):
        params[option] = float(config[section][option])

    section = 'shell_arrange'
    for option in config.options(section):
        try:
            value = config[section][option]
            params[option] = float(value)
        except ValueError:
            try:
                params[option] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                params[option] = value

    section = 'Nbody'
    for option in config.options(section):
        params[option] = config[section][option]

    section = 'ray_tracing'
    for option in config.options(section):
        try:
            params[option] = int(config[section][option])
        except ValueError:
            params[option] = config[section][option]

    return params

import re
def modify_txt_file(input_file_path, output_file_path, changes):
    with open(input_file_path, 'r') as file:
        lines = file.readlines()

    for idx, line in enumerate(lines):
        for key, value in changes.items():
            if line.startswith(key):
                match = re.search(f"{key}(\s+)", line)
                if match:
                    spaces = match.group(1)
                    prefix = line.split(key)[0] + key + spaces
                    lines[idx] = prefix + value + '\n'

    with open(output_file_path, 'w') as file:
        file.writelines(lines)