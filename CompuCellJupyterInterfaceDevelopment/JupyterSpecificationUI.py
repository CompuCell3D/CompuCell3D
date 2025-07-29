"""
Jupyter Specification UI for CompuCell3D

This module provides a comprehensive Jupyter notebook interface for configuring
CompuCell3D simulations. It includes interactive widgets for setting up:

- Metadata (processors, debug frequency)
- Potts Core parameters (dimensions, steps, neighbor order, etc.)
- Cell Types (with automatic Medium cell type management)
- Plugins (Volume, Surface, Contact, Adhesion, Chemotaxis, etc.)
- Initializers (BlobInitializer with region configuration)

The interface automatically saves configurations to JSON and provides real-time
validation of parameters. It supports both 2D and 3D simulations with
configurable boundary conditions and lattice types.

Key Features:
- Interactive tabbed interface for all simulation components
- Real-time parameter validation with error display
- Automatic saving of configurations
- Cell type management with Medium cell type handling
- Plugin-specific UI components (Volume/Surface tables, Contact energy matrix)
- BlobInitializer with multiple region support
- Integration with CompuCell3D simulation service
- Configuration validation before simulation execution
- Comprehensive error handling and troubleshooting
- Setup testing capabilities

Usage:
    # Create and display the UI
    ui = SpecificationSetupUI()

    # Test the setup before running
    ui.test_simulation_setup()

    # Run simulation with current configuration
    ui.run_and_visualize()

Dependencies:
    - ipywidgets: For interactive widgets
    - cc3d.core.PyCoreSpecs: For specification classes
    - cc3d.core.simservice.CC3DSimService: For simulation service
    - IPython.display: For widget display

Troubleshooting:
    If the simulation fails to run, use ui.test_simulation_setup() to diagnose
    issues with dependencies, configuration, or CC3D installation.

Author: CompuCell3D Development Team
License: GPL v3
"""

import os
import json
import ipywidgets as widgets
from ipywidgets import (
    VBox, HBox, Layout, Dropdown, BoundedIntText, BoundedFloatText,
    FloatText, Checkbox, Button, Text, Label, Tab, HTML, Output, GridspecLayout, Box,
    IntText, SelectMultiple, GridBox
)
from cc3d.core.PyCoreSpecs import Metadata, PottsCore, PLUGINS
from cc3d.core.PyCoreSpecs import (
    AdhesionFlexPlugin, BoundaryPixelTrackerPlugin, CellTypePlugin,
    ChemotaxisPlugin, ContactPlugin, CurvaturePlugin,
    ExternalPotentialPlugin, FocalPointPlasticityPlugin,
    LengthConstraintPlugin, PixelTrackerPlugin, SecretionPlugin,
    SurfacePlugin, VolumePlugin, BlobInitializer
)
from cc3d.core.PyCoreSpecs import SpecValueCheckError
from IPython.display import display as ipy_display

from cc3d.core.simservice.CC3DSimService import CC3DSimService

# Configuration
SAVE_FILE = 'simulation_setup.json'

def get_defaults():
    """
    Get default configuration values from class constructors.

    Returns:
        dict: Dictionary containing default values for all simulation components
              including Metadata, PottsCore, CellType, and Plugins.
    """
    return {
        "Metadata": Metadata().spec_dict,
        "PottsCore": PottsCore().spec_dict,
        "CellType": CellTypePlugin().spec_dict,
        "Plugins": {
            "AdhesionFlexPlugin": {},
            "BoundaryPixelTrackerPlugin": {},
            "ChemotaxisPlugin": {},
            "ContactPlugin": {},
            "CurvaturePlugin": {},
            "ExternalPotentialPlugin": {},
            "FocalPointPlasticityPlugin": {},
            "LengthConstraintPlugin": {},
            "PixelTrackerPlugin": {},
            "SecretionPlugin": {},
            "VolumePlugin": {},
            "SurfacePlugin": {}
        }
    }

DEFAULTS = get_defaults()

class PluginWidget:
    """
    Widget for configuring individual CompuCell3D plugins.

    This class creates interactive widgets for plugin configuration, including
    specialized UIs for VolumePlugin, SurfacePlugin, ContactPlugin, and others.
    It handles parameter validation, automatic saving, and cell type updates.

    Attributes:
        plugin_name (str): Name of the plugin (e.g., "VolumePlugin")
        plugin_class (class): The plugin class to instantiate
        default_instance: Default instance of the plugin class
        cell_types (list): List of available cell type names
        widgets (dict): Dictionary of widget components
        output (Output): Output widget for error messages
        param_cache (dict): Cache for parameter values
        parent_ui: Reference to parent UI for save operations
        potts_neighbor_order: Reference to Potts neighbor order widget
    """

    def __init__(self, plugin_name, plugin_class, saved_values, cell_types, parent_ui=None, potts_neighbor_order=None):
        """
        Initialize the plugin widget.

        Args:
            plugin_name (str): Name of the plugin
            plugin_class (class): Plugin class to configure
            saved_values (dict): Previously saved configuration values
            cell_types (list): List of available cell type names
            parent_ui: Reference to parent UI for save operations
            potts_neighbor_order: Reference to Potts neighbor order widget
        """
        self.plugin_name = plugin_name
        self.plugin_class = plugin_class
        self.default_instance = plugin_class()
        self.cell_types = cell_types
        self.widgets = {}
        self.output = Output()
        self.param_cache = {}
        self.parent_ui = parent_ui
        self.potts_neighbor_order = potts_neighbor_order
        self.create_widgets(saved_values if saved_values else {})

    def create_widgets(self, saved_values):
        """
        Create the widget components for the plugin.

        Args:
            saved_values (dict): Previously saved configuration values
        """
        # Create activation checkbox
        self.widgets["active"] = widgets.Checkbox(
            value=bool(saved_values),
            description=self.plugin_name,
            indent=False
        )

        # Create configuration container
        self.widgets["config_container"] = VBox([], layout=Layout(
            padding='0',
            display='none'
        ))

        # Setup save callback for activation toggle
        def save_on_toggle(change):
            if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
                self.parent_ui.save_to_json()
        self.widgets["active"].observe(save_on_toggle, names='value')

        # Create specialized UIs for different plugins
        if self.plugin_name == "VolumePlugin":
            self.create_volume_widgets(saved_values)
        elif self.plugin_name == "SurfacePlugin":
            self.create_surface_widgets(saved_values)
        elif self.plugin_name == "AdhesionFlexPlugin":
            self.create_adhesion_widgets(saved_values)
        elif self.plugin_name == "ContactPlugin":
            self.create_contact_widgets(saved_values)
        elif self.plugin_name == "ChemotaxisPlugin":
            self.create_chemotaxis_widgets(saved_values)
        elif self.plugin_name == "BoundaryPixelTrackerPlugin":
            self.create_boundary_tracker_widgets(saved_values)
        elif self.plugin_name in [
            "CurvaturePlugin", "ExternalPotentialPlugin", "FocalPointPlasticityPlugin",
            "LengthConstraintPlugin", "PixelTrackerPlugin", "SecretionPlugin"
        ]:
            # Not implemented plugins: message will be shown/hidden by toggle_config_visibility
            self.widgets["config_container"].children = []
            self.widgets["config_container"].layout.display = 'none'
        # else: leave as is for future plugins

        self.widgets["active"].observe(self.toggle_config_visibility, names='value')

        # Set initial visibility based on enabled state
        self.toggle_config_visibility({'new': self.widgets["active"].value})

    def toggle_config_visibility(self, change):
        """
        Toggle the visibility of configuration widgets based on activation state.

        Args:
            change (dict): Widget change event with 'new' value
        """
        # For not implemented plugins, show/hide the message based on enable state
        if self.plugin_name in [
            "CurvaturePlugin", "ExternalPotentialPlugin", "FocalPointPlasticityPlugin",
            "LengthConstraintPlugin", "PixelTrackerPlugin", "SecretionPlugin"
        ]:
            if change['new']:
                self.widgets["config_container"].children = [HTML(f"<b style='color: #b00'>{self.plugin_name} not implemented yet</b>")]
                self.widgets["config_container"].layout.display = 'block'
            else:
                self.widgets["config_container"].children = []
                self.widgets["config_container"].layout.display = 'none'
        else:
            if self.plugin_name in ["CurvaturePlugin", "ExternalPotentialPlugin", "FocalPointPlasticityPlugin"]:
                self.widgets["config_container"].layout.display = 'block'
            else:
                self.widgets["config_container"].layout.display = 'block' if change['new'] else 'none'

    def create_volume_widgets(self, saved_values):
        """
        Create widgets for VolumePlugin configuration.

        Creates a table of cell types with target volume and lambda volume parameters.
        Medium cell type is automatically disabled with zero values.

        Args:
            saved_values (dict): Previously saved volume configuration
        """
        default_params = VolumePlugin().spec_dict.get("params", [])
        default_map = {p["CellType"]: p for p in default_params} if default_params else {}
        saved_map = {p["CellType"]: p for p in saved_values.get("params", []) if "CellType" in p}
        rows = []
        for ct in self.cell_types:
            # For Medium, set target and lambda to 0 and disable
            if ct == "Medium":
                param = {"CellType": ct, "target_volume": 0.0, "lambda_volume": 0.0}
                disabled = True
            else:
                param = self.param_cache.get(ct) or saved_map.get(ct) or default_map.get(ct) or {
                    "CellType": ct,
                    "target_volume": 25.0,
                    "lambda_volume": 2.0
                }
                disabled = False

            row = {}
            row["cell_type"] = widgets.Label(value=ct, layout=widgets.Layout(width='120px'))
            row["target_volume"] = widgets.FloatText(
                value=param["target_volume"],
                min=0.0,
                description='',
                layout=widgets.Layout(width='120px'),
                disabled=disabled
            )
            row["lambda_volume"] = widgets.FloatText(
                value=param["lambda_volume"],
                min=0.0,
                description='',
                layout=widgets.Layout(width='120px'),
                disabled=disabled
            )
            def make_cache_updater(cell_type, field):
                def updater(change):
                    if cell_type not in self.param_cache:
                        self.param_cache[cell_type] = {
                            "CellType": cell_type,
                            "target_volume": row["target_volume"].value,
                            "lambda_volume": row["lambda_volume"].value
                        }
                    self.param_cache[cell_type][field] = change["new"]
                    if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
                        self.parent_ui.save_to_json()
                return updater
            row["target_volume"].observe(make_cache_updater(ct, "target_volume"), names='value')
            row["lambda_volume"].observe(make_cache_updater(ct, "lambda_volume"), names='value')
            rows.append(row)
        self.widgets["rows"] = rows
        self.update_volume_ui()

    def update_volume_ui(self):
        """Update the volume plugin UI display."""
        row_widgets = []
        for row in self.widgets["rows"]:
            row_box = HBox([
                row["cell_type"],
                widgets.Label("Target Volume:", layout=widgets.Layout(width='100px', margin='0 -5px 0 0')),  # -5px right margin
                row["target_volume"],
                widgets.HTML(value="", layout=widgets.Layout(width='10px')),  # 10px spacer
                widgets.Label("Lambda Volume:", layout=widgets.Layout(width='100px')),
                row["lambda_volume"]
            ], layout=Layout(padding='4px 0 4px 12px'))
            row_widgets.append(row_box)
        self.widgets["config_container"].children = [VBox(row_widgets)]

    def create_surface_widgets(self, saved_values):
        """
        Create widgets for SurfacePlugin configuration.

        Creates a table of cell types with target surface and lambda surface parameters.
        Medium cell type is automatically disabled with zero values.

        Args:
            saved_values (dict): Previously saved surface configuration
        """
        default_params = SurfacePlugin().spec_dict.get("params", [])
        default_map = {p["CellType"]: p for p in default_params} if default_params else {}
        saved_map = {p["CellType"]: p for p in saved_values.get("params", []) if "CellType" in p}
        rows = []
        for ct in self.cell_types:
            # For Medium, set target and lambda to 0 and disable
            if ct == "Medium":
                param = {"CellType": ct, "target_surface": 0.0, "lambda_surface": 0.0}
                disabled = True
            else:
                param = self.param_cache.get(ct) or saved_map.get(ct) or default_map.get(ct) or {
                    "CellType": ct,
                    "target_surface": 100.0,
                    "lambda_surface": 0.5
                }
                disabled = False

            row = {}
            row["cell_type"] = widgets.Label(value=ct, layout=widgets.Layout(width='120px'))
            row["target_surface"] = widgets.FloatText(
                value=param["target_surface"],
                min=0.0,
                description='',
                layout=widgets.Layout(width='120px'),
                disabled=disabled
            )
            row["lambda_surface"] = widgets.FloatText(
                value=param["lambda_surface"],
                min=0.0,
                description='',
                layout=widgets.Layout(width='120px'),
                disabled=disabled
            )
            def make_cache_updater(cell_type, field):
                def updater(change):
                    if cell_type not in self.param_cache:
                        self.param_cache[cell_type] = {
                            "CellType": cell_type,
                            "target_surface": row["target_surface"].value,
                            "lambda_surface": row["lambda_surface"].value
                        }
                    self.param_cache[cell_type][field] = change["new"]
                    if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
                        self.parent_ui.save_to_json()
                return updater
            row["target_surface"].observe(make_cache_updater(ct, "target_surface"), names='value')
            row["lambda_surface"].observe(make_cache_updater(ct, "lambda_surface"), names='value')
            rows.append(row)
        self.widgets["rows"] = rows
        self.update_surface_ui()

    def update_surface_ui(self):
        """Update the surface plugin UI display."""
        row_widgets = []
        for row in self.widgets["rows"]:
            row_box = HBox([
                row["cell_type"],
                widgets.Label("Target Surface:", layout=widgets.Layout(width='100px')),
                row["target_surface"],
                widgets.Label("Lambda Surface:", layout=widgets.Layout(width='100px')),
                row["lambda_surface"]
            ], layout=Layout(padding='4px 0 4px 12px'))
            row_widgets.append(row_box)
        self.widgets["config_container"].children = [VBox(row_widgets)]

    def create_adhesion_widgets(self, saved_values):
        """
        Create widgets for AdhesionFlexPlugin configuration.

        Uses Potts neighbor order and provides max_distance parameter.

        Args:
            saved_values (dict): Previously saved adhesion configuration
        """
        # Use Potts neighbor order instead of user input
        neighbor_order = self.potts_neighbor_order.value if self.potts_neighbor_order else 2

        # Display only - no input
        self.widgets["neighbor_order_display"] = widgets.Label(
            value=f"Neighbor Order: {neighbor_order}",
            layout=widgets.Layout(width='300px')
        )

        defaults = AdhesionFlexPlugin().spec_dict
        self.widgets["max_distance"] = widgets.IntText(
            value=saved_values.get("max_distance", defaults.get("max_distance", 3)),
            min=1,
            description='Max Distance:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["max_distance_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        max_distance_box = VBox([self.widgets["max_distance"], self.widgets["max_distance_error"]])
        container = VBox([
            self.widgets["neighbor_order_display"],
            HBox([max_distance_box])
        ])
        self.widgets["config_container"].children = [container]

    def create_contact_widgets(self, saved_values):
        """
        Create widgets for ContactPlugin configuration.

        Creates a dynamic table for contact energy parameters between cell types.
        Uses Potts neighbor order and allows adding/removing energy entries.

        Args:
            saved_values (dict): Previously saved contact configuration
        """
        from ipywidgets import VBox, HBox, Dropdown, FloatText, Button, IntText, HTML, Layout

        # Use Potts neighbor order instead of user input
        neighbor_order = self.potts_neighbor_order.value if self.potts_neighbor_order else 1

        # Display only - no input
        self.widgets["neighbor_order_display"] = widgets.Label(
            value=f"Neighbor Order: {neighbor_order}",
            layout=widgets.Layout(width='300px')
        )

        # Get cell types from the parent UI
        cell_types = self.cell_types if self.cell_types else []

        # Table for contact energies
        self.widgets["contact_rows"] = []
        self.widgets["contact_table"] = VBox()  # <-- Create this before add_row or update_table
        energies = saved_values.get("energies", {})

        # Helper to add a row
        def add_row(type_1=None, type_2=None, energy=None):
            type_1 = type_1 or (cell_types[0] if cell_types else "")
            type_2 = type_2 or (cell_types[0] if cell_types else "")
            energy = energy if energy is not None else 10.0

            dd1 = Dropdown(options=cell_types, value=type_1, layout=Layout(width='120px'))
            dd2 = Dropdown(options=cell_types, value=type_2, layout=Layout(width='120px'))
            en = FloatText(value=energy, layout=Layout(width='100px'))
            rm = Button(description="Remove", button_style='danger', layout=Layout(width='80px'))

            row = HBox([
                dd1,
                widgets.HTML(value="", layout=Layout(width='5px')),  # 5px spacer
                dd2,
                widgets.HTML(value="", layout=Layout(width='5px')),  # 5px spacer
                en,
                widgets.HTML(value="", layout=Layout(width='5px')),  # 5px spacer
                rm
            ])
            self.widgets["contact_rows"].append((row, dd1, dd2, en, rm))

            def on_remove(_):
                self.widgets["contact_rows"].remove((row, dd1, dd2, en, rm))
                update_table()
                if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
                    self.parent_ui.save_to_json()

            rm.on_click(on_remove)

            # Observe changes to trigger save
            if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
                dd1.observe(lambda change: self.parent_ui.save_to_json(), names='value')
                dd2.observe(lambda change: self.parent_ui.save_to_json(), names='value')
                en.observe(lambda change: self.parent_ui.save_to_json(), names='value')

            update_table()
            if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
                self.parent_ui.save_to_json()

        # Helper to update the table display
        def update_table():
            self.widgets["contact_table"].children = [r[0] for r in self.widgets["contact_rows"]]

        # Add from saved values
        for t1, t2dict in energies.items():
            for t2, param in t2dict.items():
                add_row(type_1=t1, type_2=t2, energy=param.get("energy", 10.0))

        # Add row button
        add_btn = Button(description="Add Row", button_style='success', layout=Layout(width='100px'))
        def on_add_btn(_):
            add_row()
            if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
                self.parent_ui.save_to_json()

        add_btn.on_click(on_add_btn)

        # Table container (already created above)
        update_table()

        # Compose the UI
        container = VBox([
            self.widgets["neighbor_order_display"],
            HTML("<b>Contact Energies</b>"),
            self.widgets["contact_table"],
            widgets.HTML(value="", layout=Layout(height='5px')),  # 5px spacer
            add_btn
        ])
        self.widgets["config_container"].children = [container]

    def create_chemotaxis_widgets(self, saved_values):
        """
        Create widgets for ChemotaxisPlugin configuration.

        Provides field name and lambda value parameters.

        Args:
            saved_values (dict): Previously saved chemotaxis configuration
        """
        defaults = ChemotaxisPlugin().spec_dict
        self.widgets["field"] = widgets.Text(
            value=saved_values.get("field", defaults.get("field", "chemoattractant")),
            description='Field Name:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='250px')
        )
        self.widgets["field_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        self.widgets["lambda_val"] = widgets.FloatText(
            value=saved_values.get("lambda", defaults.get("lambda", 100.0)),
            min=0.0,
            description='Lambda Value:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["lambda_val_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        field_box = VBox([self.widgets["field"], self.widgets["field_error"]])
        lambda_box = VBox([self.widgets["lambda_val"], self.widgets["lambda_val_error"]])
        container = VBox([
            HBox([field_box, lambda_box])
        ])
        self.widgets["config_container"].children = [container]

    def create_boundary_tracker_widgets(self, saved_values):
        """
        Create widgets for BoundaryPixelTrackerPlugin configuration.

        Uses Potts neighbor order and displays it as read-only.

        Args:
            saved_values (dict): Previously saved boundary tracker configuration
        """
        # Use Potts neighbor order instead of user input
        neighbor_order = self.potts_neighbor_order.value if self.potts_neighbor_order else 2

        # Display only - no input
        self.widgets["neighbor_order_display"] = widgets.Label(
            value=f"Neighbor Order: {neighbor_order}",
            layout=widgets.Layout(width='300px')
        )
        neighbor_box = VBox([self.widgets["neighbor_order_display"]])
        self.widgets["config_container"].children = [neighbor_box]

    def create_curvature_widgets(self, saved_values):
        """Create placeholder widgets for CurvaturePlugin (not implemented)."""
        self.widgets["config_container"].children = []
        self.widgets["config_container"].layout.display = 'block'

    def create_external_potential_widgets(self, saved_values):
        """Create placeholder widgets for ExternalPotentialPlugin (not implemented)."""
        self.widgets["config_container"].children = []
        self.widgets["config_container"].layout.display = 'block'

    def create_focal_point_plasticity_widgets(self, saved_values):
        """Create placeholder widgets for FocalPointPlasticityPlugin (not implemented)."""
        self.widgets["config_container"].children = []
        self.widgets["config_container"].layout.display = 'block'

    def create_ui(self):
        """
        Create the complete UI for the plugin.

        Returns:
            VBox: The complete plugin UI widget
        """
        if self.plugin_name in ["AdhesionFlexPlugin", "ContactPlugin", "ChemotaxisPlugin"]:
            return VBox([
                self.widgets["active"],
                self.widgets["config_container"],
                self.output
            ])
        else:
            return VBox([
                self.widgets["active"],
                self.widgets["config_container"]
            ])

    def get_config(self):
        """
        Get the current configuration for the plugin.

        Returns:
            dict or None: Plugin configuration if active, None if inactive
        """
        if not self.widgets["active"].value:
            return None

        config = {}
        if "rows" in self.widgets:
            params = []
            if self.plugin_name == "VolumePlugin":
                for row in self.widgets["rows"]:
                    params.append({
                        "CellType": row["cell_type"].value,
                        "target_volume": row["target_volume"].value,
                        "lambda_volume": row["lambda_volume"].value
                    })
                config["params"] = params
            elif self.plugin_name == "SurfacePlugin":
                for row in self.widgets["rows"]:
                    params.append({
                        "CellType": row["cell_type"].value,
                        "target_surface": row["target_surface"].value,
                        "lambda_surface": row["lambda_surface"].value
                    })
                config["params"] = params

        # ContactPlugin config
        if self.plugin_name == "ContactPlugin":
            # Use neighbor order from PottsCore
            neighbor_order = self.potts_neighbor_order.value if self.potts_neighbor_order else 1
            config = {"neighbor_order": neighbor_order, "energies": {}}
            for _, dd1, dd2, en, _ in self.widgets["contact_rows"]:
                t1, t2 = dd1.value, dd2.value
                if t1 not in config["energies"]:
                    config["energies"][t1] = {}
                config["energies"][t1][t2] = {"type_1": t1, "type_2": t2, "energy": en.value}
            return config

        for key, widget in self.widgets.items():
            if key not in ["active", "config_container", "rows", "add_btn", "contact_rows", "contact_table", "neighbor_order_display"] and hasattr(widget, 'value'):
                config[key] = widget.value

        try:
            plugin_instance = self.plugin_class()
            for k, v in config.items():
                setattr(plugin_instance, k, v)
            if self.parent_ui and hasattr(self.parent_ui, 'potts_core') and hasattr(self.parent_ui, 'cell_type_plugin'):
                plugin_instance.validate(self.parent_ui.potts_core, self.parent_ui.cell_type_plugin)
            else:
                plugin_instance.validate()
            return config
        except Exception as e:
            print(f"Validation error for {self.plugin_name}: {str(e)}")
            return None

    def reset(self):
        """Reset the plugin to default values."""
        self.widgets["active"].value = False
        self.widgets["config_container"].layout.display = 'none'
        default = self.default_instance.spec_dict

        # Clear parameter cache
        self.param_cache = {}

        if self.plugin_name == "ContactPlugin":
            # Clear contact rows and table
            self.widgets["contact_rows"] = []
            if "contact_table" in self.widgets:
                self.widgets["contact_table"].children = []
        elif "rows" in self.widgets:
            self.widgets["rows"] = []
            if "params" in default:
                if self.plugin_name == "VolumePlugin":
                    self.create_volume_widgets(default)
                elif self.plugin_name == "SurfacePlugin":
                    self.create_surface_widgets(default)

        for key, widget in self.widgets.items():
            if key != "active" and key != "config_container" and key != "rows" and key != "add_btn" and key != "contact_rows" and key != "contact_table" and key in default:
                if hasattr(widget, 'value'):
                    widget.value = default[key]

    def update_cell_types(self, cell_types):
        """
        Update cell types for the plugin.

        Args:
            cell_types (list): New list of cell type names
        """
        self.cell_types = cell_types
        if self.plugin_name == "VolumePlugin":
            self.create_volume_widgets({"params": [self.param_cache.get(ct, {"CellType": ct, "target_volume": 25.0, "lambda_volume": 2.0}) for ct in cell_types]})
        elif self.plugin_name == "SurfacePlugin":
            self.create_surface_widgets({"params": [self.param_cache.get(ct, {"CellType": ct, "target_surface": 100.0, "lambda_surface": 0.5}) for ct in cell_types]})
        elif self.plugin_name == "ContactPlugin":
            # Update dropdown options for all rows
            for row_tuple in self.widgets.get("contact_rows", []):
                _, dd1, dd2, _, _ = row_tuple
                dd1.options = cell_types
                dd2.options = cell_types

class PluginsTab:
    """
    Tab widget for managing all CompuCell3D plugins.

    This class organizes plugins into logical categories (Cell Behavior, Constraints,
    Trackers, Other Plugins) and provides a tabbed interface for easy navigation.
    It handles plugin creation, configuration, and reset functionality.

    Attributes:
        widgets (dict): Dictionary of widget components
        plugin_widgets (dict): Dictionary of PluginWidget instances
        cell_types (list): List of available cell type names
        parent_ui: Reference to parent UI for save operations
        potts_neighbor_order: Reference to Potts neighbor order widget
    """

    def __init__(self, saved_plugins, cell_types, parent_ui=None, potts_neighbor_order=None):
        """
        Initialize the plugins tab.

        Args:
            saved_plugins (dict): Previously saved plugin configurations
            cell_types (list): List of available cell type names
            parent_ui: Reference to parent UI for save operations
            potts_neighbor_order: Reference to Potts neighbor order widget
        """
        self.widgets = {}
        self.plugin_widgets = {}
        self.cell_types = cell_types
        self.parent_ui = parent_ui
        self.potts_neighbor_order = potts_neighbor_order
        self.create_widgets(saved_plugins or DEFAULTS["Plugins"])

    def create_widgets(self, saved_plugins):
        """
        Create the plugin tab widgets and organize them into categories.

        Args:
            saved_plugins (dict): Previously saved plugin configurations
        """
        self.widgets["tabs"] = widgets.Tab()
        behavior_plugins = []
        constraint_plugins = []
        tracker_plugins = []
        other_plugins = []

        plugin_classes = {
            "AdhesionFlexPlugin": AdhesionFlexPlugin,
            "BoundaryPixelTrackerPlugin": BoundaryPixelTrackerPlugin,
            "ChemotaxisPlugin": ChemotaxisPlugin,
            "ContactPlugin": ContactPlugin,
            "CurvaturePlugin": CurvaturePlugin,
            "ExternalPotentialPlugin": ExternalPotentialPlugin,
            "FocalPointPlasticityPlugin": FocalPointPlasticityPlugin,
            "LengthConstraintPlugin": LengthConstraintPlugin,
            "PixelTrackerPlugin": PixelTrackerPlugin,
            "SecretionPlugin": SecretionPlugin,
            "VolumePlugin": VolumePlugin,
            "SurfacePlugin": SurfacePlugin
        }

        for plugin_name, plugin_class in plugin_classes.items():
            plugin_values = saved_plugins.get(plugin_name, DEFAULTS["Plugins"].get(plugin_name, {}))
            plugin_widget = PluginWidget(
                plugin_name,
                plugin_class,
                plugin_values,
                self.cell_types,
                parent_ui=self.parent_ui,
                potts_neighbor_order=self.potts_neighbor_order
            )
            self.plugin_widgets[plugin_name] = plugin_widget

            if plugin_name in ["VolumePlugin", "SurfacePlugin", "LengthConstraintPlugin"]:
                constraint_plugins.append(plugin_widget.create_ui())
            elif plugin_name in ["AdhesionFlexPlugin", "ContactPlugin", "ChemotaxisPlugin",
                                "CurvaturePlugin", "ExternalPotentialPlugin",
                                "FocalPointPlasticityPlugin"]:
                behavior_plugins.append(plugin_widget.create_ui())
            elif plugin_name in ["BoundaryPixelTrackerPlugin", "PixelTrackerPlugin"]:
                tracker_plugins.append(plugin_widget.create_ui())
            else:
                other_plugins.append(plugin_widget.create_ui())

        tab_children = []
        tab_titles = []

        if behavior_plugins:
            tab_children.append(VBox(behavior_plugins))
            tab_titles.append("Cell Behavior")
        if constraint_plugins:
            tab_children.append(VBox(constraint_plugins))
            tab_titles.append("Constraints")
        if tracker_plugins:
            tab_children.append(VBox(tracker_plugins))
            tab_titles.append("Trackers")
        if other_plugins:
            tab_children.append(VBox(other_plugins))
            tab_titles.append("Other Plugins")

        self.widgets["tabs"].children = tab_children
        for i, title in enumerate(tab_titles):
            self.widgets["tabs"].set_title(i, title)

        self.widgets["reset_button"] = Button(
            description="Reset Plugins",
            button_style='warning'
        )

        # Connect reset button to save
        def on_reset_clicked(_):
            self.reset()
            if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
                self.parent_ui.save_to_json()

        self.widgets["reset_button"].on_click(on_reset_clicked)

    def get_config(self):
        """
        Get the configuration for all active plugins.

        Returns:
            dict: Dictionary of active plugin configurations
        """
        config = {}
        for plugin_name, widget in self.plugin_widgets.items():
            plugin_config = widget.get_config()
            if plugin_config is not None:
                config[plugin_name] = plugin_config
        return config

    def reset(self, _=None):
        """Reset all plugins to default values."""
        for widget in self.plugin_widgets.values():
            widget.reset()

    def update_cell_types(self, cell_types):
        """
        Update cell types for all plugins.

        Args:
            cell_types (list): New list of cell type names
        """
        self.cell_types = cell_types
        for widget in self.plugin_widgets.values():
            widget.update_cell_types(cell_types)

    def create_ui(self):
        """
        Create the complete plugins tab UI.

        Returns:
            VBox: The complete plugins tab widget
        """
        return VBox([
            self.widgets["tabs"],
            HBox([self.widgets["reset_button"]],
                 layout=Layout(justify_content='flex-start', margin='15px 0 0 0'))
        ], layout=Layout(padding='10px'))

class PottsWidget:
    """
    Widget for configuring Potts Core parameters.

    This class provides interactive widgets for all Potts Core parameters including
    dimensions, simulation steps, neighbor order, boundary conditions, and advanced
    settings like annealing and fluctuation amplitude.

    Attributes:
        widgets (dict): Dictionary of widget components
        defaults (dict): Default values for Potts Core parameters
    """

    def __init__(self, saved_values):
        """
        Initialize the Potts widget.

        Args:
            saved_values (dict): Previously saved Potts Core configuration
        """
        self.widgets = {}
        self.defaults = PottsCore().spec_dict
        # Set defaults to 1 for dimensions
        self.defaults["dim_x"] = 1
        self.defaults["dim_y"] = 1
        self.defaults["dim_z"] = 1
        self.create_widgets(saved_values or self.defaults)

    def create_widgets(self, saved_values):
        """
        Create the Potts Core parameter widgets.

        Args:
            saved_values (dict): Previously saved Potts Core configuration
        """
        # Dimension inputs with max=101
        self.widgets["dim_x"] = widgets.BoundedIntText(
            value=saved_values.get("dim_x", self.defaults["dim_x"]),
            min=1, max=101, description='X Dimension:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["dim_x_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        self.widgets["dim_y"] = widgets.BoundedIntText(
            value=saved_values.get("dim_y", self.defaults["dim_y"]),
            min=1, max=101, description='Y Dimension:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["dim_y_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        self.widgets["dim_z"] = widgets.BoundedIntText(
            value=saved_values.get("dim_z", self.defaults["dim_z"]),
            min=1, max=101, description='Z Dimension:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["dim_z_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))

        # Core parameters
        self.widgets["steps"] = widgets.IntText(
            value=saved_values.get("steps", self.defaults["steps"]),
            min=0, description='MC Steps:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["steps_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))

        # Advanced settings
        self.widgets["anneal"] = widgets.IntText(
            value=saved_values.get("anneal", self.defaults["anneal"]),
            min=0, description='Anneal Steps:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["anneal_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))

        self.widgets["fluctuation_amplitude"] = widgets.FloatText(
            value=saved_values.get("fluctuation_amplitude", self.defaults["fluctuation_amplitude"]),
            min=0.0, description='Fluctuation Amplitude:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='250px')
        )
        self.widgets["fluctuation_amplitude_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))

        # Fluctuation amplitude function
        self.widgets["fluctuation_amplitude_function"] = widgets.Dropdown(
            options=['Min', 'Max', 'Average'],
            value=saved_values.get("fluctuation_amplitude_function", self.defaults["fluctuation_amplitude_function"]),
            description='Fluctuation Function:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='250px')
        )

        self.widgets["neighbor_order"] = widgets.IntText(
            value=saved_values.get("neighbor_order", self.defaults["neighbor_order"]),
            min=1, description='Neighbor Order:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["neighbor_order_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))

        self.widgets["lattice_type"] = widgets.Dropdown(
            options=['Cartesian', 'Hexagonal'],
            value=saved_values.get("lattice_type", self.defaults["lattice_type"]),
            description='Lattice Type:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )

        # Advanced settings
        self.widgets["offset"] = widgets.FloatText(
            value=saved_values.get("offset", self.defaults.get("offset", 0.0)),
            description='Offset:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )

        # Random seed with activation checkbox
        self.widgets["use_random_seed"] = widgets.Checkbox(
            value=saved_values.get("random_seed") is not None,
            description='Use Random Seed:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["random_seed"] = widgets.BoundedIntText(
            value=saved_values.get("random_seed", 0),
            min=0,
            description='',
            disabled=not (saved_values.get("random_seed") is not None),
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )

        # Enable/disable based on checkbox
        def on_use_random_seed_change(change):
            self.widgets["random_seed"].disabled = not change.new
        self.widgets["use_random_seed"].observe(on_use_random_seed_change, names='value')

        boundary_options = ['NoFlux', 'Periodic']
        self.widgets["boundary_x"] = widgets.Dropdown(
            options=boundary_options,
            value=saved_values.get("boundary_x", self.defaults["boundary_x"]),
            description='X Boundary:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["boundary_y"] = widgets.Dropdown(
            options=boundary_options,
            value=saved_values.get("boundary_y", self.defaults["boundary_y"]),
            description='Y Boundary:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["boundary_z"] = widgets.Dropdown(
            options=boundary_options,
            value=saved_values.get("boundary_z", self.defaults["boundary_z"]),
            description='Z Boundary:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )

        # Reset button
        self.widgets["reset_button"] = Button(
            description="Reset Potts",
            button_style='warning'
        )

    def get_config(self):
        """
        Get the current Potts Core configuration.

        Returns:
            dict: Current Potts Core parameter values
        """
        return {
            "dim_x": self.widgets["dim_x"].value,
            "dim_y": self.widgets["dim_y"].value,
            "dim_z": self.widgets["dim_z"].value,
            "steps": self.widgets["steps"].value,
            "anneal": self.widgets["anneal"].value,
            "fluctuation_amplitude": self.widgets["fluctuation_amplitude"].value,
            "fluctuation_amplitude_function": self.widgets["fluctuation_amplitude_function"].value,
            "boundary_x": self.widgets["boundary_x"].value,
            "boundary_y": self.widgets["boundary_y"].value,
            "boundary_z": self.widgets["boundary_z"].value,
            "neighbor_order": self.widgets["neighbor_order"].value,
            "lattice_type": self.widgets["lattice_type"].value,
            "offset": self.widgets["offset"].value,
            "random_seed": self.widgets["random_seed"].value if self.widgets["use_random_seed"].value else None
        }

    def reset(self):
        """Reset all Potts Core parameters to default values."""
        for key, widget in self.widgets.items():
            if key != "reset_button" and key in self.defaults:
                if key == "random_seed":
                    self.widgets["use_random_seed"].value = False
                    self.widgets["random_seed"].value = 0
                else:
                    widget.value = self.defaults[key]

    def create_ui(self):
        """
        Create the complete Potts Core UI.

        Returns:
            VBox: The complete Potts Core widget
        """
        dimensions_row = HBox([
            VBox([self.widgets["dim_x"], self.widgets["dim_x_error"]]),
            widgets.HTML(value="", layout=Layout(width='5px')),  # 5px spacer
            VBox([self.widgets["dim_y"], self.widgets["dim_y_error"]]),
            widgets.HTML(value="", layout=Layout(width='5px')),  # 5px spacer
            VBox([self.widgets["dim_z"], self.widgets["dim_z_error"]])
        ], layout=Layout(justify_content='flex-start', align_items='flex-start'))
        dimensions_row.add_class('vbox-row-spacing')

        core_params_row = HBox([
            VBox([self.widgets["steps"], self.widgets["steps_error"]]),
            widgets.HTML(value="", layout=Layout(width='5px')),  # 5px spacer
            VBox([self.widgets["fluctuation_amplitude"], self.widgets["fluctuation_amplitude_error"]]),
        ], layout=Layout(justify_content='flex-start', align_items='flex-start'))
        core_params_row.add_class('vbox-row-spacing')

        core_params_row2 = HBox([
            VBox([self.widgets["neighbor_order"], self.widgets["neighbor_order_error"]]),
            widgets.HTML(value="", layout=Layout(width='5px')),  # 5px spacer
            self.widgets["lattice_type"]
        ], layout=Layout(justify_content='flex-start', align_items='flex-start'))
        core_params_row2.add_class('vbox-row-spacing')

        boundaries_row = HBox([
            self.widgets["boundary_x"],
            widgets.HTML(value="", layout=Layout(width='5px')),  # 5px spacer
            self.widgets["boundary_y"],
            widgets.HTML(value="", layout=Layout(width='5px')),  # 5px spacer
            self.widgets["boundary_z"]
        ], layout=Layout(justify_content='flex-start', align_items='flex-start'))
        boundaries_row.add_class('vbox-row-spacing')

        advanced_row1 = HBox([
            VBox([self.widgets["anneal"], self.widgets["anneal_error"]]),
            widgets.HTML(value="", layout=Layout(width='5px')),  # 5px spacer
            self.widgets["fluctuation_amplitude_function"]
        ], layout=Layout(justify_content='flex-start', align_items='flex-start'))
        advanced_row1.add_class('vbox-row-spacing')

        advanced_row2 = HBox([
            self.widgets["offset"],
            widgets.HTML(value="", layout=Layout(width='5px')),  # 5px spacer
            HBox([
                self.widgets["use_random_seed"],
                self.widgets["random_seed"]
            ], layout=Layout(justify_content='flex-start', align_items='flex-start'))
        ], layout=Layout(justify_content='flex-start', align_items='flex-start'))
        advanced_row2.add_class('vbox-row-spacing')

        return VBox([
            HTML("<b>Potts Core Parameters</b>", layout=Layout(margin='0 0 10px 0')),
            dimensions_row,
            core_params_row,
            core_params_row2,
            HTML("<b>Boundary Conditions:</b>", layout=Layout(margin='10px 0 5px 0')),
            boundaries_row,
            HTML("<b>Advanced Settings:</b>", layout=Layout(margin='10px 0 5px 0')),
            advanced_row1,
            advanced_row2,
            HBox([self.widgets["reset_button"]], layout=Layout(margin='10px 0 0 0'))
        ], layout=Layout(align_items='flex-start', padding='10px'))

class CellTypeWidget:
    """
    Widget for managing cell types in the simulation.

    This class provides an interactive interface for adding, removing, and configuring
    cell types. It automatically manages the Medium cell type (ID 0) and ensures
    proper ID assignment for new cell types. It also handles the freeze property
    for each cell type.

    Attributes:
        on_change (callable): Callback function when cell types change
        celltype_entries (list): List of cell type entry dictionaries
        widgets (dict): Dictionary of widget components
    """

    def __init__(self, saved_entries, on_change=None):
        """
        Initialize the cell type widget.

        Args:
            saved_entries (list): Previously saved cell type entries
            on_change (callable): Callback function when cell types change
        """
        self.on_change = on_change
        self.celltype_entries = []

        # Load saved entries or use defaults
        entries = saved_entries or DEFAULTS["CellType"]
        for entry in entries:
            if isinstance(entry, dict):
                self.add_entry(entry["Cell type"], entry.get("id", None), entry.get("freeze", False))
            else:
                self.add_entry(entry, None, False)

        # Ensure Medium always exists with ID 0
        if not any(entry["Cell type"] == "Medium" for entry in self.celltype_entries):
            self.add_entry("Medium", 0, False)

        self.widgets = {}
        self.create_widgets()
        self.setup_event_handlers()
        self.update_celltype_display()

    def add_entry(self, name, type_id=None, freeze=False):
        """
        Add a new cell type entry.

        Args:
            name (str): Name of the cell type
            type_id (int, optional): ID for the cell type. If None, auto-assigns
            freeze (bool): Whether the cell type should be frozen
        """
        # For Medium, force ID to 0
        if name == "Medium":
            type_id = 0
        else:
            # Find the smallest available ID
            if type_id is None:
                used_ids = {entry["id"] for entry in self.celltype_entries}
                type_id = 1
                while type_id in used_ids:
                    type_id += 1

        # Create new entry
        self.celltype_entries.append({
            "Cell type": name,
            "id": type_id,
            "freeze": freeze
        })

        # Sort by ID to maintain order
        self.celltype_entries.sort(key=lambda x: x["id"])

    def create_widgets(self):
        """Create the widget components for cell type management."""
        self.widgets["display_box"] = VBox(layout=Layout(padding='10px'))
        self.widgets["name"] = Text(
            placeholder="Cell type name",
            description="Name:",
            style={'description_width': 'initial'},
            layout=Layout(width='200px')
        )
        self.widgets["freeze"] = Checkbox(
            value=False,
            description="Freeze",
            indent=False,
            layout=Layout(width='100px')
        )
        self.widgets["add_button"] = Button(
            description="Add Cell Type",
            button_style="success",
            layout=Layout(width='120px')
        )
        self.widgets["reset_button"] = Button(
            description="Reset Cell Types",
            button_style='warning'
        )

    def setup_event_handlers(self):
        """Setup event handlers for the widgets."""
        self.widgets["add_button"].on_click(self.on_add_clicked)
        self.widgets["reset_button"].on_click(self.reset)

    def on_add_clicked(self, _):
        """Handle add button click event."""
        name = self.widgets["name"].value.strip()
        if not name:
            self.widgets["name"].placeholder = "Please enter a name!"
            return
        if any(entry["Cell type"] == name for entry in self.celltype_entries):
            self.widgets["name"].value = ""
            self.widgets["name"].placeholder = f"{name} already exists!"
            return

        self.add_entry(name, None, self.widgets["freeze"].value)
        self.update_celltype_display()
        self.widgets["name"].value = ""
        self.widgets["name"].placeholder = "Cell type name"
        if self.on_change:
            self.on_change()

    def update_celltype_display(self):
        """Update the cell type display table."""
        n = len(self.celltype_entries)
        if n == 0:
            self.widgets["display_box"].children = [HTML("<i>No cell types defined.</i>")]
            return

        row_border = '1px solid #e0e0e0'
        header_border = '2px solid #bdbdbd'
        header = [
            HTML(f"<b style='display:block; padding:2px 8px;'>ID</b>", layout=Layout(border=f'0 0 {header_border} 0')),
            HTML(f"<b style='display:block; padding:2px 8px;'>Cell Type</b>", layout=Layout(border=f'0 0 {header_border} 0')),
            HTML(f"<b style='display:block; padding:2px 8px;'>Freeze</b>", layout=Layout(border=f'0 0 {header_border} 0')),
            HTML(f"<b style='display:block; padding:2px 8px;'>Remove</b>", layout=Layout(border=f'0 0 {header_border} 0'))
        ]
        grid = GridspecLayout(n + 1, 4, grid_gap="0px")
        for j, h in enumerate(header):
            grid[0, j] = h

        for i, entry in enumerate(self.celltype_entries):
            border_style = f'0 0 {row_border} 0' if i < n - 1 else '0'

            # ID column
            grid[i + 1, 0] = Label(str(entry['id']), layout=Layout(border=border_style, padding='2px 8px'))

            # Cell Type column
            grid[i + 1, 1] = Label(str(entry['Cell type']), layout=Layout(border=border_style, padding='2px 8px'))

            # Freeze column
            freeze_chk = Checkbox(
                value=entry.get('freeze', False),
                indent=False,
                layout=Layout(border=border_style, padding='2px 8px', width='50px')
            )

            # Handler for freeze checkbox
            def make_freeze_handler(idx):
                def handler(change):
                    self.celltype_entries[idx]['freeze'] = change['new']
                    if self.on_change:
                        self.on_change()
                return handler

            freeze_chk.observe(make_freeze_handler(i), names='value')
            grid[i + 1, 2] = freeze_chk

            # Remove column - disable for Medium
            if entry['Cell type'] == "Medium":
                remove_btn = Button(
                    description="Default",
                    button_style='',
                    disabled=True,
                    layout=Layout(width='100px', border=border_style, padding='2px 8px')
                )
            else:
                remove_btn = Button(
                    description="Remove",
                    button_style='danger',
                    layout=Layout(width='100px', border=border_style, padding='2px 8px')
                )
                # Handler for remove button
                def make_remove_handler(idx):
                    def handler(_):
                        # Skip Medium removal
                        if self.celltype_entries[idx]['Cell type'] == "Medium":
                            return
                        del self.celltype_entries[idx]
                        self.update_celltype_display()
                        if self.on_change:
                            self.on_change()
                    return handler
                remove_btn.on_click(make_remove_handler(i))

            grid[i + 1, 3] = remove_btn

        self.widgets["display_box"].children = [grid]

    def get_config(self):
        """
        Get the current cell type configuration.

        Returns:
            list: List of cell type entry dictionaries
        """
        return self.celltype_entries.copy()

    def get_cell_type_names(self):
        """
        Get the list of cell type names.

        Returns:
            list: List of cell type names
        """
        return [entry["Cell type"] for entry in self.celltype_entries]

    def reset(self, _=None):
        """Reset cell types to default (Medium only)."""
        self.celltype_entries = []
        self.add_entry("Medium", 0, False)
        self.update_celltype_display()

    def create_ui(self):
        """
        Create the complete cell type UI.

        Returns:
            VBox: The complete cell type widget
        """
        input_row = HBox([
            self.widgets["name"],
            widgets.HTML(value="", layout=Layout(width='10px')),  # 10px spacer
            self.widgets["freeze"],
            widgets.HTML(value="", layout=Layout(width='10px')),  # 10px spacer
            self.widgets["add_button"]
        ], layout=Layout(justify_content='flex-start', margin='10px 0'))

        reset_button_box = HBox([self.widgets["reset_button"]],
                               layout=Layout(justify_content='flex-start', margin='10px 0'))

        return VBox([
            HTML("<b>Cell Types</b>", layout=Layout(margin='0 0 10px 0')),
            self.widgets["display_box"],
            input_row,
            reset_button_box
        ], layout=Layout(align_items='flex-start', padding='10px'))

class InitializerWidget:
    """
    Widget for configuring simulation initializers.

    This class provides an interface for configuring different types of initializers,
    currently supporting BlobInitializer with multiple region configuration.
    It allows users to define regions with specific dimensions, positions, and
    cell type assignments.

    Attributes:
        cell_types (list): List of available cell type names
        saved_values (dict): Previously saved initializer configuration
        widgets (dict): Dictionary of widget components
        regions (list): List of region configuration dictionaries
        regions_box (VBox): Container for region widgets
        add_region_btn (Button): Button to add new regions
        parent_ui: Reference to parent UI for save operations
        initializer_type_dropdown (Dropdown): Dropdown for initializer type selection
    """

    def __init__(self, saved_values=None, cell_types=None, parent_ui=None):
        """
        Initialize the initializer widget.

        Args:
            saved_values (dict): Previously saved initializer configuration
            cell_types (list): List of available cell type names
            parent_ui: Reference to parent UI for save operations
        """
        self.cell_types = cell_types or []
        self.saved_values = saved_values or {}
        self.widgets = {}
        self.regions = []  # List of region dicts
        self.regions_box = VBox()
        self.add_region_btn = Button(description="Add Region", button_style="success")
        self.add_region_btn.on_click(self.add_region)
        self.parent_ui = parent_ui

        # Dropdown for initializer type
        self.initializer_type_dropdown = Dropdown(
            options=["BlobInitializer", "PIFInitializer", "UniformInitializer"],
            value=self.saved_values.get("type", "BlobInitializer"),
            description="Initializer Type:",
            style={'description_width': 'initial'},
        )
        self.initializer_type_dropdown.observe(self._on_initializer_type_change, names='value')

        # Load regions from saved_values if present and type is BlobInitializer
        if self.saved_values.get("type", "BlobInitializer") == "BlobInitializer" and self.saved_values.get("regions"):
            for region in self.saved_values["regions"]:
                self._add_region_from_saved(region)

        self.create_ui()

    def _on_initializer_type_change(self, change):
        """
        Handle initializer type change.

        Args:
            change (dict): Widget change event
        """
        self.update_regions_box()
        if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
            self.parent_ui.save_to_json()

    def _add_region_from_saved(self, region):
        """
        Add a region from saved configuration.

        Args:
            region (dict): Saved region configuration
        """
        region_dict = {
            "width": IntText(value=region["width"], description="Width:"),
            "radius": IntText(value=region["radius"], description="Radius:"),
            "center_x": IntText(value=region["center"][0], description="Center X:", layout=Layout(width='225px')),
            "center_y": IntText(value=region["center"][1], description="Center Y:", layout=Layout(width='225px')),
            "center_z": IntText(value=region["center"][2], description="Center Z:", layout=Layout(width='225px')),
            "cell_types": SelectMultiple(
                options=self.cell_types,
                value=tuple(region["cell_types"]),
                description="Cell Types:"
            ),
            "remove_btn": Button(description="Remove", button_style="danger"),
            "selection_note": HTML(
                value="<medium><em>💡 Tip: Hold Ctrl/Cmd to select multiple cell types</em></medium>",
                layout=Layout(margin='0 0 0 10px')
            )
        }
        region_dict["remove_btn"].on_click(lambda btn, r=region_dict: self.remove_region(r))
        for key in ["width", "radius", "center_x", "center_y", "center_z"]:
            region_dict[key].observe(self._trigger_save, names='value')
        region_dict["cell_types"].observe(self._trigger_save, names='value')
        self.regions.append(region_dict)

    def _trigger_save(self, *_):
        """Trigger save operation."""
        if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
            self.parent_ui.save_to_json()

    def add_region(self, _=None):
        """Add a new region to the initializer."""
        width = 5  # Manually assigned default value
        radius = 20
        center_x = 50
        center_y = 50
        center_z = 0
        selected_cell_types = self.cell_types.copy() if self.cell_types else []

        region = {
            "width": IntText(value=width, description="Width:"),
            "radius": IntText(value=radius, description="Radius:"),
            "center_x": IntText(value=center_x, description="Center X:", layout=Layout(width='225px')),
            "center_y": IntText(value=center_y, description="Center Y:", layout=Layout(width='225px')),
            "center_z": IntText(value=center_z, description="Center Z:", layout=Layout(width='225px')),
            "cell_types": SelectMultiple(
                options=self.cell_types,
                value=tuple(selected_cell_types),
                description="Cell Types:"
            ),
            "remove_btn": Button(description="Remove", button_style="danger"),
            # Add note about multi-selection
            "selection_note": HTML(
                value="<small><em>💡 Tip: Hold Ctrl/Cmd to select multiple cell types</em></small>",
                layout=Layout(margin='0 0 0 10px')
            )
        }

        region["remove_btn"].on_click(lambda btn, r=region: self.remove_region(r))

        for key in ["width", "radius", "center_x", "center_y", "center_z"]:
            region[key].observe(self._trigger_save, names='value')

        region["cell_types"].observe(self._trigger_save, names='value')

        self.regions.append(region)
        self.update_regions_box()
        self._trigger_save()

    def remove_region(self, region):
        """
        Remove a region from the initializer.

        Args:
            region (dict): Region to remove
        """
        self.regions.remove(region)
        self.update_regions_box()
        self._trigger_save()

    def update_regions_box(self):
        """Update the regions display based on initializer type."""
        selected_type = self.initializer_type_dropdown.value
        if selected_type == "BlobInitializer":
            region_vboxes = []
            for region in self.regions:
                vbox = VBox([
                    HBox([region["center_x"], region["center_y"], region["center_z"]]),
                    HBox([region["width"], region["radius"]]),
                    VBox([
                        region["cell_types"],
                        region["selection_note"]
                    ]),
                    region["remove_btn"]
                ], layout=Layout(border="1px solid #ccc", margin="10px 0", padding="10px"))
                region_vboxes.append(vbox)
            self.regions_box.children = region_vboxes + [self.add_region_btn]
        else:
            msg = f"{selected_type} not implemented yet"
            self.regions_box.children = [HTML(f"<b style='color: #b00'>{msg}</b>")]

    def get_config(self):
        """
        Get the current initializer configuration.

        Returns:
            dict: Current initializer configuration
        """
        selected_type = self.initializer_type_dropdown.value
        if selected_type == "BlobInitializer":
            regions = []
            for region in self.regions:
                regions.append({
                    "width": region["width"].value,
                    "radius": region["radius"].value,
                    "center": (region["center_x"].value, region["center_y"].value, region["center_z"].value),
                    "cell_types": list(region["cell_types"].value)
                })
            return {
                "type": "BlobInitializer",
                "regions": regions
            }
        else:
            return {
                "type": selected_type
            }

    def create_ui(self):
        """Create the initializer UI."""
        self.update_regions_box()
        self.widget = VBox([
            self.initializer_type_dropdown,
            Label("Initializer Configuration:"),
            self.regions_box
        ])

    def update_cell_types(self, cell_types):
        """
        Update cell types for the initializer.

        Args:
            cell_types (list): New list of cell type names
        """
        self.cell_types = cell_types
        for region in self.regions:
            region["cell_types"].options = self.cell_types

    def get_widget(self):
        """
        Get the initializer widget.

        Returns:
            VBox: The initializer widget
        """
        return self.widget

class SpecificationSetupUI:
    """
    Main UI class for CompuCell3D simulation specification setup.

    This class provides a comprehensive Jupyter notebook interface for configuring
    all aspects of a CompuCell3D simulation. It includes tabs for Metadata,
    Potts Core parameters, Cell Types, Plugins, Initializers, and Steppables.

    The interface automatically saves configurations to JSON and provides real-time
    validation. It supports both 2D and 3D simulations with configurable
    boundary conditions and lattice types.

    Key Features:
    - Interactive tabbed interface for all simulation components
    - Real-time parameter validation with error display
    - Automatic saving of configurations to JSON
    - Cell type management with Medium cell type handling
    - Plugin-specific UI components (Volume/Surface tables, Contact energy matrix)
    - BlobInitializer with multiple region support
    - Integration with CompuCell3D simulation service
    - Configuration validation before simulation execution
    - Comprehensive error handling and troubleshooting
    - Setup testing capabilities

    Attributes:
        _initializing (bool): Flag to suppress save_to_json during initialization
        widgets (dict): Dictionary of widget components
        saved_values (dict): Previously saved configuration values
        metadata (Metadata): Metadata specification object
        potts_core (PottsCore): Potts Core specification object
        cell_type_plugin (CellTypePlugin): Cell Type plugin specification object
        potts_widget (PottsWidget): Widget for Potts Core configuration
        celltype_widget (CellTypeWidget): Widget for cell type management
        plugins_tab (PluginsTab): Tab widget for plugin configuration
        initializer_widget (InitializerWidget): Widget for initializer configuration
        cc3d_sim: CompuCell3D simulation service instance
        visualization_output (Output): Output widget for simulation visualization
    """

    def __init__(self):
        """
        Initialize the specification setup UI.

        Creates all widget components, loads saved values, and sets up
        the complete interface with all tabs and functionality.
        """
        self._initializing = True  # Flag to suppress save_to_json during init
        self.widgets = {}
        self.saved_values = self.load_saved_values()
        self.metadata = Metadata()
        self.potts_core = PottsCore()
        self.cell_type_plugin = CellTypePlugin()
        self.apply_saved_values()
        self.create_metadata_widgets()
        self.potts_widget = PottsWidget(self.saved_values.get("PottsCore"))
        self.celltype_widget = CellTypeWidget(
            self.saved_values.get("CellType"),
            on_change=self.cell_types_changed
        )
        self.plugins_tab = PluginsTab(
            self.saved_values.get("Plugins", {}),
            self.celltype_widget.get_cell_type_names(),
            parent_ui=self,
            potts_neighbor_order=self.potts_widget.widgets["neighbor_order"]
        )
        self.initializer_widget = InitializerWidget(
            saved_values=self.saved_values.get("Initializer"),
            cell_types=self.celltype_widget.get_cell_type_names(),
            parent_ui=self
        )
        self.cc3d_sim = None
        self.visualization_output = widgets.Output()
        self.create_ui()
        self.setup_event_handlers()
        self._initializing = False  # Allow save_to_json after init

    def print_specs_summary(self):
        """
        Print summary of loaded specs for debugging.

        Displays current configuration for Metadata, PottsCore, Cell Types,
        active plugins, and initializer settings.
        """
        print("Loaded Specifications:")
        print(f"Metadata: {self.metadata.spec_dict}")
        print(f"PottsCore: {self.potts_core.spec_dict}")
        print(f"Cell Types: {[ct['Cell type'] for ct in self.celltype_widget.get_config()]}")

        active_plugins = [name for name, config in self.plugins_tab.get_config().items() if config]
        print(f"Active Plugins: {active_plugins}")

        init_config = self.initializer_widget.get_config()
        print(f"Initializer: {init_config['type']} with {len(init_config.get('regions', []))} regions")

    def validate_configuration(self):
        """
        Validate the current configuration before running simulation.

        Returns:
            tuple: (is_valid, error_messages) where is_valid is a boolean
                   and error_messages is a list of error strings
        """
        errors = []

        # Check Metadata
        try:
            metadata_config = self.metadata.spec_dict
            if metadata_config.get("num_processors", 1) < 1:
                errors.append("Number of processors must be at least 1")
        except Exception as e:
            errors.append(f"Metadata validation error: {e}")

        # Check PottsCore
        try:
            potts_config = self.potts_core.spec_dict
            if potts_config.get("dim_x", 1) < 1 or potts_config.get("dim_y", 1) < 1 or potts_config.get("dim_z", 1) < 1:
                errors.append("All dimensions must be at least 1")
            if potts_config.get("steps", 0) < 0:
                errors.append("MC Steps must be non-negative")
            if potts_config.get("neighbor_order", 1) < 1:
                errors.append("Neighbor order must be at least 1")
        except Exception as e:
            errors.append(f"PottsCore validation error: {e}")

        # Check Cell Types
        try:
            cell_types = self.celltype_widget.get_cell_type_names()
            if not cell_types:
                errors.append("At least one cell type must be defined")
            if "Medium" not in cell_types:
                errors.append("Medium cell type must be present")
        except Exception as e:
            errors.append(f"Cell types validation error: {e}")

        # Check Plugins
        try:
            plugins_config = self.plugins_tab.get_config()
            for plugin_name, plugin_config in plugins_config.items():
                if plugin_config is None:
                    continue
                # Add specific plugin validation here if needed
        except Exception as e:
            errors.append(f"Plugins validation error: {e}")

        # Check Initializer
        try:
            init_config = self.initializer_widget.get_config()
            if init_config.get("type") == "BlobInitializer":
                regions = init_config.get("regions", [])
                if not regions:
                    errors.append("BlobInitializer requires at least one region")
                for i, region in enumerate(regions):
                    if region.get("width", 0) <= 0:
                        errors.append(f"Region {i+1}: width must be positive")
                    if region.get("radius", 0) <= 0:
                        errors.append(f"Region {i+1}: radius must be positive")
                    if not region.get("cell_types"):
                        errors.append(f"Region {i+1}: must specify at least one cell type")
        except Exception as e:
            errors.append(f"Initializer validation error: {e}")

        return len(errors) == 0, errors

    @property
    def specs(self):
        """
        Returns a list of all specification objects for the simulation.

        This includes Metadata, PottsCore, CellTypePlugin, enabled plugins,
        and initializer. The specs are used to configure the CompuCell3D
        simulation service.

        Returns:
            list: List of specification objects for the simulation
        """
        import copy
        import inspect
        print("Generating simulation specifications...")

        try:
            config = self.current_config()
            specs = []

            # Metadata
            specs.append(Metadata(**config["Metadata"]))

            # PottsCore (filter out invalid keys)
            pottscore_args = inspect.signature(PottsCore.__init__).parameters
            pottscore_config = {k: v for k, v in config["PottsCore"].items() if k in pottscore_args}
            specs.append(PottsCore(**pottscore_config))

            # CellTypePlugin
            cell_types = [entry["Cell type"] for entry in config["CellType"]]
            specs.append(CellTypePlugin(*cell_types))

            # VolumePlugin
            if "params" in config["Plugins"].get("VolumePlugin", {}):
                volume_plugin = VolumePlugin()
                for param in config["Plugins"]["VolumePlugin"]["params"]:
                    volume_plugin.param_new(param["CellType"], param["target_volume"], param["lambda_volume"])
                specs.append(volume_plugin)

            # SurfacePlugin
            if "params" in config["Plugins"].get("SurfacePlugin", {}):
                surface_plugin = SurfacePlugin()
                for param in config["Plugins"]["SurfacePlugin"]["params"]:
                    surface_plugin.param_new(param["CellType"], param["target_surface"], param["lambda_surface"])
                specs.append(surface_plugin)

            # ContactPlugin
            if "energies" in config["Plugins"].get("ContactPlugin", {}):
                cp_conf = config["Plugins"]["ContactPlugin"]
                contact_plugin = ContactPlugin(cp_conf.get("neighbor_order", 1))
                for t1, t2dict in cp_conf["energies"].items():
                    for t2, param in t2dict.items():
                        contact_plugin.param_new(type_1=param["type_1"], type_2=param["type_2"], energy=param["energy"])
                specs.append(contact_plugin)

            # Other plugins
            for plugin_name in ["AdhesionFlexPlugin", "BoundaryPixelTrackerPlugin", "ChemotaxisPlugin"]:
                if plugin_name in config["Plugins"] and config["Plugins"][plugin_name]:
                    plugin_class = globals()[plugin_name]
                    plugin_config = config["Plugins"][plugin_name]
                    plugin_args = inspect.signature(plugin_class.__init__).parameters
                    filtered_config = {k: v for k, v in plugin_config.items() if k in plugin_args}
                    specs.append(plugin_class(**filtered_config))

            # BlobInitializer
            if "Initializer" in config and config["Initializer"].get("type") == "BlobInitializer":
                blob_init = BlobInitializer()
                for region in config["Initializer"].get("regions", []):
                    blob_init.region_new(
                        width=region["width"],
                        radius=region["radius"],
                        center=region["center"],
                        cell_types=region["cell_types"]
                    )
                specs.append(blob_init)

            print(f"Generated {len(specs)} specification objects")
            return specs

        except Exception as e:
            print(f"Error generating specifications: {e}")
            import traceback
            traceback.print_exc()
            return []

    def cell_types_changed(self):
        """
        Handle cell type changes.

        Updates plugin cell types and saves configuration when cell types
        are added, removed, or modified.
        """
        self.update_plugin_cell_types()
        self.save_to_json()

    def update_plugin_cell_types(self):
        """
        Update cell types for all plugins and initializer.

        Propagates cell type changes to VolumePlugin, SurfacePlugin,
        ContactPlugin, and BlobInitializer widgets.
        """
        cell_types = self.celltype_widget.get_config()
        cell_type_names = [entry["Cell type"] for entry in cell_types]
        # Update VolumePlugin
        volume_widget = self.plugins_tab.plugin_widgets.get("VolumePlugin")
        if volume_widget:
            volume_widget.update_cell_types(cell_type_names)
        # Update SurfacePlugin
        surface_widget = self.plugins_tab.plugin_widgets.get("SurfacePlugin")
        if surface_widget:
            surface_widget.update_cell_types(cell_type_names)
        # Update ContactPlugin
        contact_widget = self.plugins_tab.plugin_widgets.get("ContactPlugin")
        if contact_widget:
            contact_widget.update_cell_types(cell_type_names)
        # Update BlobInitializer
        if hasattr(self.initializer_widget, "update_cell_types"):
            self.initializer_widget.update_cell_types(cell_type_names)

    def apply_saved_values(self):
        """
        Apply saved values to specification objects.

        Loads previously saved configuration values into the Metadata,
        PottsCore, and CellTypePlugin objects.
        """
        if "Metadata" in self.saved_values:
            for key, value in self.saved_values["Metadata"].items():
                if hasattr(self.metadata, key):
                    setattr(self.metadata, key, value)
        if "PottsCore" in self.saved_values:
            for key, value in self.saved_values["PottsCore"].items():
                if hasattr(self.potts_core, key):
                    setattr(self.potts_core, key, value)
        if "CellType" in self.saved_values:
            for entry in self.saved_values["CellType"]:
                if isinstance(entry, dict):
                    cell_type_name = entry["Cell type"]
                    if cell_type_name == "Medium":
                        continue
                    self.cell_type_plugin.cell_type_append(
                        cell_type_name,
                        frozen=entry.get("freeze", False)
                    )

    def create_metadata_widgets(self):
        """Create widgets for metadata configuration."""
        self.widgets["num_processors"] = widgets.IntText(
            value=self.metadata.num_processors,
            min=1,
            description='Number of Processors:',
            style={'description_width': 'initial'}
        )
        self.widgets["num_processors_error"] = HTML(
            value="",
            layout=Layout(
                margin='2px 0 5px 0',
                display='none'
            )
        )
        self.widgets["debug_output_frequency"] = widgets.IntText(
            value=self.metadata.debug_output_frequency,
            min=0,
            description='Debug Output Frequency:',
            style={'description_width': 'initial'}
        )
        self.widgets["debug_output_frequency_error"] = HTML(
            value="",
            layout=Layout(
                margin='2px 0 5px 0',
                display='none'
            )
        )
        self.widgets["reset_metadata_button"] = Button(
            description="Reset Metadata",
            button_style='warning'
        )

    def load_saved_values(self):
        """
        Load saved configuration values from JSON file.

        Returns:
            dict: Saved configuration values or defaults if file doesn't exist
        """
        try:
            if os.path.exists(SAVE_FILE):
                with open(SAVE_FILE, 'r') as f:
                    return json.load(f)
            return json.loads(json.dumps(DEFAULTS))
        except (json.JSONDecodeError, IOError):
            return json.loads(json.dumps(DEFAULTS))

    def current_config(self):
        """
        Get the current configuration from all widgets.

        Returns:
            dict: Complete current configuration including all components
        """
        return {
            "Metadata": self.metadata.spec_dict,
            "PottsCore": self.potts_core.spec_dict,
            "CellType": self.celltype_widget.get_config(),
            "Plugins": self.plugins_tab.get_config(),
            "Initializer": self.initializer_widget.get_config()
        }

    def save_to_json(self, _=None):
        """
        Save current configuration to JSON file.

        Saves the complete configuration to the SAVE_FILE location.
        Suppressed during initialization to prevent premature saves.
        """
        if getattr(self, '_initializing', False):
            return  # Don't save during initialization
        config = self.current_config()
        if config:
            with open(SAVE_FILE, 'w') as f:
                json.dump(config, f, indent=4)

    def setup_event_handlers(self):
        """Setup event handlers for all widgets."""
        self.widgets["num_processors"].observe(
            lambda change: self.update_metadata('num_processors', change.new),
            names='value'
        )
        self.widgets["debug_output_frequency"].observe(
            lambda change: self.update_metadata('debug_output_frequency', change.new),
            names='value'
        )
        self.potts_widget.widgets["reset_button"].on_click(
            lambda _: self.reset_potts_tab()
        )
        self.celltype_widget.widgets["reset_button"].on_click(
            lambda _: self.reset_celltype_tab()
        )
        self.widgets["reset_metadata_button"].on_click(
            lambda _: self.reset_metadata_tab()
        )
        for name, widget in self.potts_widget.widgets.items():
            if hasattr(widget, 'observe') and name != "reset_button":
                widget.observe(
                    lambda change, prop=name: self.update_potts_core(prop, change.new),
                    names='value'
                )
        self.celltype_widget.widgets["add_button"].on_click(
            lambda _: self.update_cell_types()
        )
        for widget in [self.widgets["num_processors"], self.widgets["debug_output_frequency"]]:
            widget.observe(lambda _: self.save_to_json(), names='value')

    def update_metadata(self, property_name, value):
        """
        Update metadata property and handle validation.

        Args:
            property_name (str): Name of the property to update
            value: New value for the property
        """
        try:
            setattr(self.metadata, property_name, value)
            self.save_to_json()
            self.clear_metadata_error(property_name)
        except Exception as e:
            self.show_metadata_error(property_name, str(e))

    def show_metadata_error(self, property_name, message):
        """
        Show error message for metadata property.

        Args:
            property_name (str): Name of the property with error
            message (str): Error message to display
        """
        error_widget_name = f"{property_name}_error"
        input_widget = self.widgets.get(property_name)
        if error_widget_name in self.widgets:
            error_widget = self.widgets[error_widget_name]
            error_widget.value = f'<span style="color: red; font-size: 12px;">⚠️ {message}</span>'
            error_widget.layout.display = 'block'
        if input_widget is not None and hasattr(input_widget, 'add_class'):
            input_widget.add_class('error-input')

    def clear_metadata_error(self, property_name):
        """
        Clear error message for metadata property.

        Args:
            property_name (str): Name of the property to clear error for
        """
        error_widget_name = f"{property_name}_error"
        input_widget = self.widgets.get(property_name)
        if error_widget_name in self.widgets:
            error_widget = self.widgets[error_widget_name]
            error_widget.value = ""
            error_widget.layout.display = 'none'
        if input_widget is not None and hasattr(input_widget, 'remove_class'):
            input_widget.remove_class('error-input')

    def update_potts_core(self, property_name, value):
        """
        Update Potts Core property and handle validation.

        Args:
            property_name (str): Name of the property to update
            value: New value for the property
        """
        try:
            setattr(self.potts_core, property_name, value)
            self.save_to_json()
            self.clear_potts_error(property_name)
        except Exception as e:
            self.show_potts_error(property_name, str(e))

    def show_potts_error(self, property_name, message):
        """
        Show error message for Potts Core property.

        Args:
            property_name (str): Name of the property with error
            message (str): Error message to display
        """
        error_widget_name = f"{property_name}_error"
        input_widget = self.potts_widget.widgets.get(property_name)
        if error_widget_name in self.potts_widget.widgets:
            error_widget = self.potts_widget.widgets[error_widget_name]
            error_widget.value = f'<span style="color: red; font-size: 12px;">⚠️ {message}</span>'
            error_widget.layout.display = 'block'
        if input_widget is not None and hasattr(input_widget, 'add_class'):
            input_widget.add_class('error-input')

    def clear_potts_error(self, property_name):
        """
        Clear error message for Potts Core property.

        Args:
            property_name (str): Name of the property to clear error for
        """
        error_widget_name = f"{property_name}_error"
        input_widget = self.potts_widget.widgets.get(property_name)
        if error_widget_name in self.potts_widget.widgets:
            error_widget = self.potts_widget.widgets[error_widget_name]
            error_widget.value = ""
            error_widget.layout.display = 'none'
        if input_widget is not None and hasattr(input_widget, 'remove_class'):
            input_widget.remove_class('error-input')

    def run_and_visualize(self, _=None):
        """
        Run the simulation and create visualization.

        This method initializes the CompuCell3D simulation service with the current
        configuration, starts the simulation, and creates interactive visualization
        widgets for real-time monitoring. It follows the same pattern as the working
        examples in the CompuCell3D demo notebooks.

        The method performs the following steps:
        1. Validates the current configuration
        2. Generates simulation specifications from UI settings
        3. Initializes and starts the CC3DSimService
        4. Creates visualization widgets for real-time monitoring
        5. Displays run/pause controls for simulation interaction

        **Prerequisites:**
        - All required simulation parameters must be configured in the UI
        - At least one cell type must be defined (Medium is automatically included)
        - Valid initializer configuration must be present
        - CompuCell3D must be properly installed with simservice support

        **Parameters:**
            _ (Any, optional): Unused parameter for widget callback compatibility.
                              Defaults to None.

        **Returns:**
            None: This method does not return a value but displays widgets directly.

        **Raises:**
            ImportError: If CC3DSimService cannot be imported, indicating missing
                       CompuCell3D installation or simservice support.
            Exception: Various exceptions may be raised during simulation setup,
                      with detailed error messages and troubleshooting tips provided.

        **Examples:**
            # Basic usage in a Jupyter notebook
            ui = SpecificationSetupUI()
            ui.run_and_visualize()

            # The method will display:
            # - Validation status messages
            # - Simulation initialization progress
            # - Visualization widget (if available)
            # - Run/pause control button

        **Troubleshooting:**
            If the method fails, check the following:
            1. Configuration validation errors - fix parameter values in the UI
            2. Missing cell types - ensure at least Medium and one other cell type
            3. Invalid initializer settings - check region parameters
            4. CompuCell3D installation - verify simservice support is available

        **Widget Output:**
            The method displays several widgets in the notebook:
            - Visualization widget showing the simulation state
            - Run/pause toggle button for controlling simulation execution
            - Status messages indicating simulation progress
            - Error messages if validation or setup fails

        **Notes:**
            - The simulation starts in a paused state by default
            - Use the run button to start/pause simulation execution
            - Visualization updates automatically as the simulation progresses
            - The method handles both successful and failed visualization creation
            - Manual step/stop controls are provided as fallback if run button fails
        """
        from cc3d.core.simservice.CC3DSimService import CC3DSimService
        from IPython.display import display
        import traceback
        import time

        try:
            print("Validating configuration...")
            is_valid, errors = self.validate_configuration()

            if not is_valid:
                print("❌ Configuration validation failed:")
                for error in errors:
                    print(f"  - {error}")
                print("\nPlease fix the configuration errors before running the simulation.")
                return

            print("✅ Configuration validation passed!")
            print("Initializing CompuCell3D simulation...")

            # Get current configuration
            specs = self.specs

            if not specs:
                print("❌ Failed to generate simulation specifications")
                print("Please check your configuration and try again.")
                return

            print(f"Configuration includes {len(specs)} specification objects")

            # Initialize simulation service
            self.cc3d_sim = CC3DSimService()

            # Register specifications
            self.cc3d_sim.register_specs(specs)

            print("Specifications registered successfully")

            # Run the simulation (this compiles and prepares everything)
            self.cc3d_sim.run()
            print("Simulation compiled and prepared")

            # Initialize the simulation
            self.cc3d_sim.init()
            print("Simulation initialized")

            # Start the simulation
            self.cc3d_sim.start()
            print("Simulation started successfully")

            print("Creating visualization...")

            # Create visualization widget - follow the working pattern from the notebook
            try:
                viewer = self.cc3d_sim.visualize().show()
                print("Visualization created successfully")
            except Exception as e:
                print(f"Warning: Error creating visualization: {e}")
                print("Creating basic status widget...")
                # Create a simple status widget instead
                status_widget = widgets.HTML(
                    value="<div style='padding: 10px; border: 1px solid #ccc; background: #f9f9f9;'>"
                          "<h3>Simulation Status</h3>"
                          "<p>✅ Simulation is running</p>"
                          "<p>Current Step: <span id='step'>0</span></p>"
                          "<p>Use the run button below to control simulation</p>"
                          "</div>"
                )
                display(status_widget)

            # Try to create run button - follow the working pattern from the notebook
            try:
                run_button = self.cc3d_sim.jupyter_run_button()
                if run_button:
                    display(run_button)
                    print("Run button created - use it to pause/resume simulation")
                else:
                    print("Run button not available")
            except (AttributeError, RuntimeError) as e:
                print(f"Run button not available: {e}")
                # Create a simple manual control
                manual_controls = widgets.HBox([
                    widgets.Button(description="Step", button_style='info'),
                    widgets.Button(description="Stop", button_style='danger')
                ])

                def on_step(b):
                    try:
                        self.cc3d_sim.step()
                        print(f"Stepped to step {self.cc3d_sim.current_step}")
                    except Exception as e:
                        print(f"Error stepping simulation: {e}")

                def on_stop(b):
                    try:
                        self.cc3d_sim.stop()
                        print("Simulation stopped")
                    except Exception as e:
                        print(f"Error stopping simulation: {e}")

                manual_controls.children[0].on_click(on_step)
                manual_controls.children[1].on_click(on_stop)
                display(manual_controls)

            print("Simulation setup complete!")
            print("The simulation is now running. Use the controls above to interact with it.")

        except ImportError as e:
            print(f"Error importing CC3DSimService: {e}")
            print("Please ensure CompuCell3D is properly installed with simservice support")
            traceback.print_exc()
        except Exception as e:
            print(f"Error during simulation setup: {e}")
            print("Full error details:")
            traceback.print_exc()
            print("\nTroubleshooting tips:")
            print("1. Check that all required plugins are properly configured")
            print("2. Verify that cell types are correctly defined")
            print("3. Ensure that initializer settings are valid")
            print("4. Check that Potts Core parameters are within valid ranges")

    def create_ui(self):
        """
        Create the complete UI with all tabs and components.

        This method constructs the full Jupyter notebook interface for CompuCell3D
        simulation configuration. It creates a tabbed interface with all necessary
        components for setting up and running simulations, including configuration
        panels, validation, and execution controls.

        The UI consists of the following components:
        1. **Metadata Tab**: Global simulation settings (processors, debug frequency)
        2. **Potts Core Tab**: Lattice dimensions, neighbor order, boundary conditions
        3. **Cell Types Tab**: Cell type management with automatic Medium cell type
              handling and freeze options
        4. **Plugins Tab**: Plugin configuration organized by category
        5. **Initializer Tab**: Simulation initialization settings
        6. **Steppable Tab**: Steppable configuration (work in progress)
        7. **Run Container**: Execution controls and visualization output

        **UI Structure:**
            The interface is organized as follows:
            ```
            ┌─────────────────────────────────────────────────────────┐
            │                    Tabbed Interface                    │
            ├─────────────────────────────────────────────────────────┤
            │ Metadata │ Potts Core │ Cell Types │ Plugins │ ...    │
            ├─────────────────────────────────────────────────────────┤
            │                                                       │
            │              Tab Content Area                         │
            │                                                       │
            ├─────────────────────────────────────────────────────────┤
            │              Run Simulation Button                    │
            │              Visualization Output                      │
            └─────────────────────────────────────────────────────────┘
            ```

        **Tab Descriptions:**
            - **Metadata**: Configure global simulation parameters like number of
              processors and debug output frequency
            - **Potts Core**: Set lattice dimensions, simulation steps, neighbor
              order, boundary conditions, and advanced settings
            - **Cell Types**: Manage cell types with automatic Medium cell type
              handling and freeze options
            - **Plugins**: Configure simulation plugins organized into categories:
              - Cell Behavior: Adhesion, Contact, Chemotaxis, etc.
              - Constraints: Volume, Surface, Length constraints
              - Trackers: Boundary pixel tracking
              - Other Plugins: Additional functionality
            - **Initializer**: Set up simulation initialization with support for
              BlobInitializer with multiple regions
            - **Steppable**: Steppable configuration (currently placeholder)

        **Interactive Features:**
            - Real-time parameter validation with error display
            - Automatic saving of configurations to JSON
            - Reset buttons for individual tabs and global reset
            - Dynamic cell type management with automatic updates
            - Plugin-specific UI components (tables, matrices, etc.)
            - Integration with CompuCell3D simulation service

        **Parameters:**
            None: This method takes no parameters.

        **Returns:**
            None: This method does not return a value but displays the UI directly
                  using IPython.display.display().

        **Examples:**
            # Basic usage in a Jupyter notebook
            ui = SpecificationSetupUI()
            # The UI is automatically displayed when the class is instantiated

            # Accessing individual components
            metadata_widget = ui.widgets["num_processors"]
            potts_widget = ui.potts_widget
            cell_types = ui.celltype_widget

        **Configuration Persistence:**
            The UI automatically saves configurations to 'simulation_setup.json'
            and loads them on initialization. This ensures that user settings
            persist between notebook sessions.

        **Validation:**
            The UI includes comprehensive validation for all parameters:
            - Dimension constraints (1-101 for lattice dimensions)
            - Cell type requirements (Medium must be present)
            - Plugin parameter validation
            - Initializer region validation

        **Styling:**
            The UI uses custom CSS classes for consistent styling:
            - Rounded corners for input widgets
            - Error state styling for invalid inputs
            - Responsive layout with proper spacing
            - Color-coded buttons and status indicators

        **Integration:**
            The UI integrates with:
            - CompuCell3D simulation service for execution
            - Jupyter widgets for interactive components
            - JSON file system for configuration persistence
            - CC3D core specifications for validation

        **Notes:**
            - The UI is designed to be self-contained and user-friendly
            - All changes are automatically saved to prevent data loss
            - Error messages provide clear guidance for fixing issues
            - The interface supports both 2D and 3D simulations
            - Advanced features are organized in logical categories
        """
        tabs = Tab(layout=Layout(width='800px'))

        # Create tab containers
        metadata_tab = VBox([
            self.create_metadata_tab()
        ], layout=Layout(width='750px'))

        potts_tab = VBox([
            self.potts_widget.create_ui()
        ], layout=Layout(width='750px'))

        celltype_tab = VBox([
            self.celltype_widget.create_ui()
        ], layout=Layout(width='750px'))

        plugins_tab = VBox([
            self.plugins_tab.create_ui()
        ], layout=Layout(width='750px'))

        initializer_tab = VBox([
            self.initializer_widget.get_widget()
        ], layout=Layout(width='750px'))

        steppable_tab = VBox([
            HTML("<h3>Steppable Configuration</h3>"),
            HTML("<b style='color: #b00'>Steppable configuration is work in progress.</b style='color: #b00'>"),
            HTML("<b style='color: #b00'>This feature will be implemented in a future release.</b style='color: #b00'>")
        ], layout=Layout(width='750px'), padding='15px')

        tabs.children = [metadata_tab, potts_tab, celltype_tab, plugins_tab, initializer_tab, steppable_tab]
        tabs.set_title(0, 'Metadata')
        tabs.set_title(1, 'Potts Core')
        tabs.set_title(2, 'Cell Types')
        tabs.set_title(3, 'Plugins')
        tabs.set_title(4, 'Initializer')
        tabs.set_title(5, 'Steppable')

        # Wrap in container for consistent styling
        container = VBox([
            tabs
        ], layout=Layout(
            align_items='flex-start',
            width='850px',
            padding='15px')
        )
        ipy_display(container)

    def create_metadata_tab(self):
        """
        Create the metadata tab UI.

        Returns:
            VBox: The metadata tab widget
        """
        num_processors_box = VBox([
            self.widgets["num_processors"],
            self.widgets["num_processors_error"]
        ], layout=Layout(align_items='flex-start'))

        debug_frequency_box = VBox([
            self.widgets["debug_output_frequency"],
            self.widgets["debug_output_frequency_error"]
        ], layout=Layout(align_items='flex-start'))

        return VBox([
            HTML("<b>Simulation Metadata</b>", layout=Layout(margin='0 0 10px 0')),
            num_processors_box,
            debug_frequency_box,
            HBox([self.widgets["reset_metadata_button"]], layout=Layout(justify_content='flex-start', margin='10px 0'))
        ], layout=Layout(align_items='flex-start', padding='10px'))

    def reset_potts_tab(self):
        """Reset Potts Core tab to default values."""
        self.potts_widget.reset()
        for prop, value in PottsCore().spec_dict.items():
            if hasattr(self.potts_core, prop):
                setattr(self.potts_core, prop, value)
        self.save_to_json()

    def reset_celltype_tab(self):
        """Reset Cell Types tab to default values."""
        self.celltype_widget.reset()
        self.cell_type_plugin = CellTypePlugin()
        self.update_plugin_cell_types()
        self.save_to_json()

    def reset_plugins_tab(self):
        """Reset Plugins tab to default values."""
        self.plugins_tab = PluginsTab(
            DEFAULTS["Plugins"],
            self.celltype_widget.get_cell_type_names(),
            parent_ui=self,
            potts_neighbor_order=self.potts_widget.widgets["neighbor_order"]
        )
        self.save_to_json()

    def reset_metadata_tab(self):
        """Reset Metadata tab to default values."""
        self.widgets["num_processors"].value = Metadata().num_processors
        self.widgets["debug_output_frequency"].value = Metadata().debug_output_frequency
        self.metadata = Metadata()
        self.save_to_json()

    def update_cell_types(self):
        """
        Update cell types in the cell type plugin.

        Rebuilds the cell type plugin with current cell type entries
        and updates all dependent widgets.
        """
        self.cell_type_plugin = CellTypePlugin()
        for entry in self.celltype_widget.celltype_entries:
            self.cell_type_plugin.cell_type_append(
                entry["Cell type"],
                frozen=entry["freeze"]
            )
        self.plugins_tab.update_cell_types(self.celltype_widget.get_cell_type_names())
        self.save_to_json()

    def test_simulation_setup(self):
        """
        Test the simulation setup to verify dependencies and configuration.

        This method checks if all required components are available and
        validates the current configuration without actually running a simulation.
        """
        self.visualization_output.clear_output()
        with self.visualization_output:
            print("🧪 Testing CompuCell3D simulation setup...")
            print("=" * 50)

            # Test 1: Check imports
            print("1. Checking imports...")
            try:
                from cc3d.core.simservice.CC3DSimService import CC3DSimService
                print("   ✅ CC3DSimService import successful")
            except ImportError as e:
                print(f"   ❌ CC3DSimService import failed: {e}")
                print("   This may indicate that simservice is not properly installed")
                return

            try:
                from cc3d.core.PyCoreSpecs import Metadata, PottsCore, CellTypePlugin
                print("   ✅ Core specs imports successful")
            except ImportError as e:
                print(f"   ❌ Core specs import failed: {e}")
                return

            # Test 2: Check configuration validation
            print("\n2. Validating configuration...")
            is_valid, errors = self.validate_configuration()
            if is_valid:
                print("   ✅ Configuration validation passed")
            else:
                print("   ❌ Configuration validation failed:")
                for error in errors:
                    print(f"      - {error}")

            # Test 3: Check specification generation
            print("\n3. Testing specification generation...")
            try:
                specs = self.specs
                if specs:
                    print(f"   ✅ Generated {len(specs)} specification objects")
                    for i, spec in enumerate(specs):
                        print(f"      {i+1}. {type(spec).__name__}")
                else:
                    print("   ❌ No specifications generated")
            except Exception as e:
                print(f"   ❌ Specification generation failed: {e}")

            # Test 4: Check CC3D availability
            print("\n4. Checking CC3D availability...")
            try:
                test_sim = CC3DSimService()
                print("   ✅ CC3DSimService instantiation successful")

                # Test if we can create a minimal simulation
                try:
                    from cc3d.core.PyCoreSpecs import Metadata, PottsCore, CellTypePlugin
                    test_specs = [Metadata(), PottsCore(), CellTypePlugin("Medium")]
                    test_sim.register_specs(test_specs)
                    print("   ✅ Basic specification registration successful")
                except Exception as e:
                    print(f"   ⚠️  Basic specification registration failed: {e}")

            except Exception as e:
                print(f"   ❌ CC3DSimService instantiation failed: {e}")

            # Test 5: Check visualization capabilities
            print("\n5. Checking visualization capabilities...")
            try:
                test_sim = CC3DSimService()
                has_visualize = hasattr(test_sim, 'visualize')
                has_run_button = hasattr(test_sim, 'jupyter_run_button')

                if has_visualize:
                    print("   ✅ visualize() method available")
                else:
                    print("   ⚠️  visualize() method not available")

                if has_run_button:
                    print("   ✅ jupyter_run_button() method available")
                else:
                    print("   ⚠️  jupyter_run_button() method not available")

            except Exception as e:
                print(f"   ❌ Visualization capability check failed: {e}")

            print("\n" + "=" * 50)
            print("🏁 Setup test complete!")

            if is_valid:
                print("✅ Your configuration appears to be ready for simulation")
                print("You can now try running the simulation with ui.run_and_visualize()")
            else:
                print("❌ Configuration issues detected")
                print("Please fix the validation errors before running the simulation")

class SimulationVisualizer:
    """
    Class for visualizing CompuCell3D simulations.

    This class provides methods for creating initial state visualizations
    and running live simulations with real-time visualization.

    Attributes:
        specs (list): List of specification objects for the simulation
        cc3d_sim: CompuCell3D simulation service instance
    """

    def __init__(self, specs):
        """
        Initialize the simulation visualizer.

        Args:
            specs (list): List of specification objects for the simulation
        """
        self.specs = specs
        self.cc3d_sim = None

    def show_initializer(self):
        """
        Visualize the initial cell configuration.

        Creates a temporary simulation to generate the initial state
        and displays it without running the full simulation.

        Returns:
            viewer: Visualization widget showing initial state
        """
        # Create temporary simulation to generate initial state
        temp_sim = CC3DSimService()
        temp_sim.register_specs(self.specs)
        temp_sim.run()
        temp_sim.init()  # Initialize but don't start simulation

        # Visualize at step 0 (initial state)
        viewer = temp_sim.visualize(plot_freq=1)
        viewer.show()  # Fixed: use show() instead of draw()
        return viewer

    def show_simulation(self):
        """
        Run and visualize the live simulation.

        Creates and starts the simulation service, then creates a
        visualization widget for real-time monitoring.

        Returns:
            viewer: Visualization widget for live simulation
        """
        # Create and start simulation service
        self.cc3d_sim = CC3DSimService()
        self.cc3d_sim.register_specs(self.specs)
        self.cc3d_sim.run()
        self.cc3d_sim.init()
        self.cc3d_sim.start()

        # Create visualization widget
        viewer = self.cc3d_sim.visualize(plot_freq=10)
        viewer.show()  # Fixed: use show() instead of draw()

        # Add simulation controls
        run_button = self.cc3d_sim.jupyter_run_button()
        if run_button:
            display(run_button)

        return viewer

    def stop_simulation(self):
        """Stop the running simulation."""
        if self.cc3d_sim:
            self.cc3d_sim.stop()
            self.cc3d_sim = None