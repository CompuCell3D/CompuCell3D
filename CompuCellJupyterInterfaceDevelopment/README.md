# CompuCell3D Jupyter Widget

An interactive Jupyter widget for configuring CompuCell3D simulations specification through a visual interface.

## Overview
This project provides a comprehensive Jupyter Notebook interface for setting up CompuCell3D simulations. It features:
- Interactive tabs for configuring simulation parameters
- Visual controls for cell types, physics parameters, and plugins
- Live validation and error handling
- Simulation visualization within Jupyter
- Configuration persistence through JSON saving/loading

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Future Development](#future-development)
- [Contributors](#contributors)
- [License](#license)

## Features
- **Metadata Configuration**: Set processors and debug output
- **Potts Core**: Configure dimensions, boundaries, and physics
- **Cell Types**: Define cell types with freezing capability
- **Plugins**:
  - Volume/Surface constraints
  - Adhesion/Contact energies
  - Chemotaxis/Boundary tracking
- **Initializers**: Blob initialization with region controls
- **Visualization**: Integrated simulation viewer

## Installation
### Step 1: Install CompuCell3D
Download the appropriate package for your OS:
   [CompuCell3D installation](https://compucell3d.org/SrcBin)

### Step 2: Step Up a Python Environment
# Create virtual environment (recommended)
```
python -m venv cc3d_env
source cc3d_env/bin/activate  # Linux/macOS
cc3d_env\Scripts\activate    # Windows

# Install Python dependencies
pip install ipywidgets>=8.0.0 jupyter>=1.0.0 numpy>=1.21.0
```

### Step 3:  Install the Widget
```
git clone (Github link)
cd cc3d-jupyter-widget
jupyter notebook
```

## Usage
```python
from JupyterSpecificationUI import SpecificationSetupUI
from JupyterWidgetStyling import inject_jupyter_widget_css

# Widget Styling
inject_jupyter_widget_css()

# Initialize and display the UI
ui = SpecificationSetupUI()

# Run Visualization
ui.run_and_visualize()
```

For detailed information about simulation methods and advanced usage, see the [Simulation Methods Documentation](Notes%20for%20Developer/Simulation_Methods_Documentation.md).

## Future Development
* Implement additional plugins (Curvature, FocalPointPlasticity, etc.)
* Add PIF/Uniform initializers
* Steppable configuration
* Enhanced visualization controls

## Contributors
* Dr. T.J. Sego, 2025
* Steve Han, 2025
* Jinyao Huang, 2025

## License

This project is licensed under the MIT License - see the [LICENSE](License.txt) file for details.
