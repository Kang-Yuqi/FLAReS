import configparser
import ast
import os 
from . import initial_power_spectrum
from . import gen_Gadget
from . import plot_shell_arrange
from . import box2dens
from . import analytical_power_spectrum
from . import nbody_analysis
from . import nG_analysis
from . import needlets
from . import shell_thickness_corr
from . import lensing_post_Born

def read_config(filepath):

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not exists")

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(filepath)

    params = {}

    sections = {
        'path': str,
        'cosmology_model': float,
        'shell_arrange': None,
        'Nbody': None,
        'shell_projection': None,
        'lensing': None,
        'analysis': None
    }

    for section, dtype in sections.items():
        if config.has_section(section):
            for option in config.options(section):
                value = config[section][option]
                if dtype == float:
                    params[option] = float(value)
                elif dtype == str:
                    params[option] = value
                else:
                    try:
                        params[option] = int(value)
                    except ValueError:
                        try:
                            params[option] = ast.literal_eval(value)
                        except (ValueError, SyntaxError):
                            params[option] = value
        else:
            print(f"Warning: Section '{section}' not found.")

    return params
