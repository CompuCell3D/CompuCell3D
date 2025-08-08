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
- [Quick Start](#quick-start)
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

## Quick Start

Go to **[SortingExample.ipynb](SortingExample.ipynb)** to access a complete cell sorting simulation example demonstrating all major features including 2D simulation setup, cell type configuration, volume constraints, contact energies, and live visualization.

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
