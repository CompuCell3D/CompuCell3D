#!/usr/bin/env python3
"""
Script to fix the import issue in the SortingExample.ipynb notebook.
Replaces the problematic cc3d imports with direct path imports.
"""

import json
import os

def fix_notebook_import():
    """Fix the import issue in the notebook."""
    notebook_path = "CompuCell3D/core/Demos/CC3DPy/notebooks/SortingExample.ipynb"

    # Read the notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)

    # Find the cell with the problematic imports
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            if 'from cc3d.core.j_spec.JupyterWidgetStyling import inject_jupyter_widget_css' in source:
                # Replace the problematic import
                new_source = [
                    "# Optional Styling for Jupyter Widgets\n",
                    "# This code injects CSS styles for Jupyter widgets to enhance their appearance.\n",
                    "import sys\n",
                    "import os\n",
                    "# Add the cc3d directory to the Python path\n",
                    "cc3d_path = r'B:/projects/AT3/Keke_CompuCell3D_fork/cc3d'\n",
                    "sys.path.insert(0, cc3d_path)\n",
                    "from core.j_spec.JupyterWidgetStyling import inject_jupyter_widget_css\n",
                    "\n",
                    "inject_jupyter_widget_css()\n"
                ]
                cell['source'] = new_source
                print("Fixed the JupyterWidgetStyling import in the notebook")
            elif 'from cc3d.core.j_spec.JupyterSpecificationUI import SpecificationSetupUI' in source:
                # Replace the problematic import
                new_source = [
                    "# Create the UI and register the specifications\n",
                    "# This will allow the user to interactively set up the simulation parameters\n",
                    "import sys\n",
                    "import os\n",
                    "# Add the cc3d directory to the Python path\n",
                    "cc3d_path = r'B:/projects/AT3/Keke_CompuCell3D_fork/cc3d'\n",
                    "sys.path.insert(0, cc3d_path)\n",
                    "from core.j_spec.JupyterSpecificationUI import SpecificationSetupUI\n",
                    "\n",
                    "ui = SpecificationSetupUI()\n"
                ]
                cell['source'] = new_source
                print("Fixed the JupyterSpecificationUI import in the notebook")

    # Write the fixed notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=1)

    print(f"Updated {notebook_path}")

if __name__ == "__main__":
    fix_notebook_import()
