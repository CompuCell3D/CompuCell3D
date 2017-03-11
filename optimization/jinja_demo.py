import os
from jinja2 import Environment, FileSystemLoader
from collections import OrderedDict

# Capture our current directory


def print_rendered_xml():
    # Create the jinja2 environment.
    # Notice the use of trim_blocks, which greatly helps control whitespace.

    THIS_DIR = os.path.dirname(os.path.abspath(__file__))

    j2_env = Environment(loader=FileSystemLoader(THIS_DIR),
                         trim_blocks=True)

    param_labels = ['TEMPERATURE', 'CONTACT_A_B', 'CONTACT_B_B']
    param_vals = [12.2,13.0, 20]

    param_dict = OrderedDict( zip(param_labels,param_vals ))
    print  'param_dict=',param_dict


    print j2_env.get_template('short_demo.xml').render(
        **param_dict
    )


if __name__ == '__main__':
    print_rendered_xml()