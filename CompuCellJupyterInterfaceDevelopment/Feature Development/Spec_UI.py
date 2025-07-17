import os
import json
import ipywidgets as widgets
from ipywidgets import (
    VBox, HBox, Layout, Dropdown, BoundedIntText, BoundedFloatText,
    FloatText, Checkbox, Button, Text, Label, Tab, HTML, Output, GridspecLayout, Box
)
from IPython.display import display, HTML as IPythonHTML
from cc3d.core.PyCoreSpecs import Metadata, PottsCore, PLUGINS
from cc3d.core.PyCoreSpecs import (
    AdhesionFlexPlugin, BoundaryPixelTrackerPlugin, CellTypePlugin,
    ChemotaxisPlugin, ContactPlugin, CurvaturePlugin,
    ExternalPotentialPlugin, FocalPointPlasticityPlugin,
    LengthConstraintPlugin, PixelTrackerPlugin, SecretionPlugin,
    SurfacePlugin, VolumePlugin
)
from cc3d.core.PyCoreSpecs import SpecValueCheckError

# Inject CSS for input styling and error states
display(IPythonHTML("""
<style>
/* Round corners for all input boxes */
.widget-text input,
.widget-bounded-int-text input,
.widget-bounded-float-text input,
.widget-float-text input,
.widget-int-text input {
    border-radius: 4px !important;
}

/* Round corners for dropdown/select inputs */
.widget-dropdown select,
.widget-select select {
    border-radius: 4px !important;
}

/* Round corners for buttons */
.widget-button button,
.jupyter-button {
    border-radius: 4px !important;
}

/* Spacing classes for layout containers */
.vbox-row-spacing {
    margin: 10px 0 !important;
}

.hbox-item-spacing {
    margin: 0 15px 0 0 !important;
}

.vbox-no-margin {
    margin: 0 !important;
}

.hbox-no-margin {
    margin: 0 !important;
}

.celltype-item-spacing {
    margin: 0 0 5px 0 !important;
}

.small-right-spacing {
    margin: 0 5px 0 0 !important;
}

/* Plugin-specific spacing classes */
.plugin-config-container {
    padding: 0 0 0 0 !important;
}
.plugin-compact-container {
    padding: 0 !important;
    margin: 0 !important;
}
.plugin-input-spacing {
    margin: 0 15px 0 0 !important;
}
.plugin-bottom-spacing {
    margin: 0 0 15px 0 !important;
}
.plugin-checkbox-bottom-spacing {
    margin: 0 0 15px 0 !important;
}
.button-spacing {
    margin: 10px 0 !important;
}

/* Error state styling with rounded corners */
.error-input input {
    border: 2px solid #f44336 !important;
    background-color: #ffebee !important;
    box-shadow: 0 0 3px rgba(244, 67, 54, 0.3) !important;
    border-radius: 4px !important;
}

.error-input input:focus {
    border-color: #d32f2f !important;
    box-shadow: 0 0 5px rgba(244, 67, 54, 0.5) !important;
    border-radius: 4px !important;
}

.plugin-top-spacing {
    margin-top: 10px !important;
}
</style>
"""))

# Configuration
SAVE_FILE = 'simulation_setup.json'

# Get default values from class constructors using .spec_dict
def get_defaults():
    return {
        "Metadata": Metadata().spec_dict,
        "PottsCore": PottsCore().spec_dict,
        # "CellType": [{"Cell type": "Medium", "id": 0, "freeze": False}], # shouldn't be manually set ID automatically defined.
        "CellType": [{"Cell type": "Medium", "freeze": False}], # shouldn't be manually set ID automatically defined.
        "Plugins": {
            # All plugins default to empty dict (unchecked)
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
    def __init__(self, plugin_name, plugin_class, saved_values, cell_types, parent_ui=None):
        self.plugin_name = plugin_name
        self.plugin_class = plugin_class
        self.default_instance = plugin_class()
        self.cell_types = cell_types
        self.widgets = {}
        self.output = Output()  # Add output widget for debug
        self.param_cache = {}  # For VolumePlugin: cache values by cell type
        self.parent_ui = parent_ui  # <-- add this line
        # Only use saved_values if it is not empty, otherwise use empty dict
        self.create_widgets(saved_values if saved_values else {})

    def create_widgets(self, saved_values):
        # Main checkbox with plugin name
        self.widgets["active"] = widgets.Checkbox(
            value=bool(saved_values),
            description=self.plugin_name,
            indent=False
        )

        # Container for plugin-specific widgets
        self.widgets["config_container"] = VBox([], layout=Layout(
            padding='0',
            display='none' if not saved_values else 'block'
        ))

        # Save to JSON when checkbox is toggled
        def save_on_toggle(change):
            print(f"[DEBUG] Checkbox toggled for {self.plugin_name}, new value: {change['new']}")
            if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
                print("[DEBUG] Calling save_to_json from checkbox toggle")
                self.parent_ui.save_to_json()
        self.widgets["active"].observe(save_on_toggle, names='value')

        # Create plugin-specific widgets
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
        elif self.plugin_name == "CurvaturePlugin":
            self.create_curvature_widgets(saved_values)
        elif self.plugin_name == "ExternalPotentialPlugin":
            self.create_external_potential_widgets(saved_values)
        elif self.plugin_name == "FocalPointPlasticityPlugin":
            self.create_focal_point_plasticity_widgets(saved_values)
        # Add other plugins as needed...

        # Set up active toggle
        self.widgets["active"].observe(self.toggle_config_visibility, names='value')

    def toggle_config_visibility(self, change):
        # For plugins without input boxes, always show the config_container for spacing
        if self.plugin_name in ["CurvaturePlugin", "ExternalPotentialPlugin", "FocalPointPlasticityPlugin"]:
            self.widgets["config_container"].layout.display = 'block'
        else:
            self.widgets["config_container"].layout.display = 'block' if change['new'] else 'none'

    # VolumePlugin widgets
    def create_volume_widgets(self, saved_values):
        """Widgets for VolumePlugin"""
        # Build a row for every cell type in self.cell_types
        # Use saved values if present, else default
        default_params = VolumePlugin().spec_dict.get("params", [])
        default_map = {p["CellType"]: p for p in default_params} if default_params else {}
        # Build a map from cell type to saved param
        saved_map = {p["CellType"]: p for p in saved_values.get("params", []) if "CellType" in p}
        # Use cached values if present, else saved, else default
        rows = []
        for ct in self.cell_types:
            # Priority: cache > saved > default > fallback
            param = self.param_cache.get(ct) or saved_map.get(ct) or default_map.get(ct) or {
                "CellType": ct,
                "target_volume": 25.0,
                "lambda_volume": 2.0
            }
            row = {}
            row["cell_type"] = widgets.Label(value=ct, layout=widgets.Layout(width='120px'))
            row["target_volume"] = widgets.FloatText(
                value=param["target_volume"],
                min=1.0,
                description='',
                layout=widgets.Layout(width='120px')
            )
            row["lambda_volume"] = widgets.FloatText(
                value=param["lambda_volume"],
                min=0.0,
                description='',
                layout=widgets.Layout(width='120px')
            )
            # Update cache and save on change
            def make_cache_updater(cell_type, field):
                def updater(change):
                    if cell_type not in self.param_cache:
                        self.param_cache[cell_type] = {
                            "CellType": cell_type,
                            "target_volume": row["target_volume"].value,
                            "lambda_volume": row["lambda_volume"].value
                        }
                    self.param_cache[cell_type][field] = change["new"]
                    # Save to JSON immediately
                    try:
                        from IPython.core.getipython import get_ipython
                        shell = get_ipython()
                        if shell and hasattr(shell, 'user_ns') and 'ui' in shell.user_ns:
                            ui = shell.user_ns['ui']
                            if hasattr(ui, 'save_to_json'):
                                ui.save_to_json()
                    except ImportError:
                        pass
                return updater
            row["target_volume"].observe(make_cache_updater(ct, "target_volume"), names='value')
            row["lambda_volume"].observe(make_cache_updater(ct, "lambda_volume"), names='value')
            rows.append(row)
        self.widgets["rows"] = rows
        self.update_volume_ui()

    def update_volume_ui(self):
        """Update the UI after row changes"""
        # Show a row for every cell type
        row_widgets = []
        for row in self.widgets["rows"]:
            row_box = HBox([
                row["cell_type"],
                widgets.Label("Target Volume:", layout=widgets.Layout(width='100px')),
                row["target_volume"],
                widgets.Label("Lambda Volume:", layout=widgets.Layout(width='100px')),
                row["lambda_volume"]
            ], layout=Layout(padding='4px 0 4px 12px'))
            row_box.add_class('volume-row')
            row_widgets.append(row_box)
        self.widgets["config_container"].children = [VBox(row_widgets)]

    # SurfacePlugin widgets
    def create_surface_widgets(self, saved_values):
        """Widgets for SurfacePlugin"""
        params = saved_values.get("params", [])

        if not params:
            default_params = SurfacePlugin().spec_dict.get("params", [])
            params = default_params or [{
                "CellType": "Medium",
                "target_surface": 100.0,
                "lambda_surface": 0.5
            }]

        rows = []
        for param in params:
            cell_type_value = param["CellType"]
            if cell_type_value not in self.cell_types:
                cell_type_value = self.cell_types[0] if self.cell_types else "Medium"

            row = {}
            row["cell_type"] = widgets.Dropdown(
                options=self.cell_types,
                value=cell_type_value,
                description='Cell Type:',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='200px')
            )
            row["target_surface"] = widgets.FloatText(
                value=param["target_surface"],
                min=1.0,
                description='Target Surface:',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='200px')
            )
            row["lambda_surface"] = widgets.FloatText(
                value=param["lambda_surface"],
                min=0.0,
                description='Lambda Surface:',
                style={'description_width': 'initial'},
                layout=widgets.Layout(width='200px')
            )
            rows.append(row)

        self.widgets["rows"] = rows

        # Add button for new row
        self.widgets["add_btn"] = widgets.Button(
            description="Add Constraint",
            button_style='success'
        )

        # Define add_surface_row as a method
        def add_surface_row(_):
            default_params = SurfacePlugin().spec_dict.get("params", [])
            default_values = default_params[0] if default_params else {
                "CellType": "Medium",
                "target_surface": 100.0,
                "lambda_surface": 0.5
            }

            new_row = {
                "cell_type": widgets.Dropdown(
                    options=self.cell_types,
                    value=default_values["CellType"],
                    description='Cell Type:',
                    style={'description_width': 'initial'}
                ),
                "target_surface": widgets.FloatText(
                    value=default_values["target_surface"],
                    min=1.0,
                    description='Target Surface:',
                    style={'description_width': 'initial'}
                ),
                "lambda_surface": widgets.FloatText(
                    value=default_values["lambda_surface"],
                    min=0.0,
                    description='Lambda Surface:',
                    style={'description_width': 'initial'}
                )
            }

            self.widgets["rows"].append(new_row)
            self.update_surface_ui()

        # Connect the button handler
        self.widgets["add_btn"].on_click(add_surface_row)

        # Build UI
        self.update_surface_ui()

    def update_surface_ui(self):
        """Update the UI after row changes"""
        row_widgets = []
        for i, row in enumerate(self.widgets["rows"]):
            # Add spacing classes
            row["cell_type"].add_class('plugin-input-spacing')
            row["target_surface"].add_class('plugin-input-spacing')
            row["lambda_surface"].add_class('plugin-input-spacing')
            remove_btn = widgets.Button(
                description="Remove",
                button_style='danger',
                layout=Layout(width='100px')
            )

            def make_remove_handler(index):
                def handler(_):
                    del self.widgets["rows"][index]
                    self.update_surface_ui()
                return handler

            remove_btn.on_click(make_remove_handler(i))

            row_widgets.append(HBox([
                row["cell_type"],
                row["target_surface"],
                row["lambda_surface"],
                remove_btn
            ]))

        self.widgets["config_container"].children = [
            *row_widgets,
            self.widgets["add_btn"]
        ]

    # AdhesionFlexPlugin widgets
    def create_adhesion_widgets(self, saved_values):
        """Widgets for AdhesionFlexPlugin"""
        defaults = AdhesionFlexPlugin().spec_dict
        # Input widgets
        self.widgets["neighbor_order"] = widgets.IntText(
            value=saved_values.get("neighbor_order", defaults.get("neighbor_order", 2)),
            description='Neighbor Order:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["neighbor_order"].add_class('plugin-input-spacing')
        self.widgets["neighbor_order_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        self.widgets["max_distance"] = widgets.IntText(
            value=saved_values.get("max_distance", defaults.get("max_distance", 3)),
            description='Max Distance:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["max_distance_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        # UI: error under each input
        neighbor_box = VBox([self.widgets["neighbor_order"], self.widgets["neighbor_order_error"]])
        max_distance_box = VBox([self.widgets["max_distance"], self.widgets["max_distance_error"]])
        container = VBox([
            HBox([neighbor_box, max_distance_box])
        ])
        container.add_class('plugin-compact-container')
        self.widgets["config_container"].children = [container]
        self.widgets["config_container"].add_class('plugin-top-spacing')
        self.widgets["config_container"].add_class('plugin-bottom-spacing')
        self.widgets["neighbor_order"].observe(lambda change: self._validate_plugin_input('AdhesionFlexPlugin', 'neighbor_order', change.new), names='value')
        self.widgets["max_distance"].observe(lambda change: self._validate_plugin_input('AdhesionFlexPlugin', 'max_distance', change.new), names='value')
        self.widgets["neighbor_order"].observe(self._on_adhesionflex_input_change, names='value')
        self.widgets["max_distance"].observe(self._on_adhesionflex_input_change, names='value')

    # ContactPlugin widgets
    def create_contact_widgets(self, saved_values):
        """Widgets for ContactPlugin"""
        defaults = ContactPlugin().spec_dict
        self.widgets["neighbor_order"] = widgets.IntText(
            value=saved_values.get("neighbor_order", defaults.get("neighbor_order", 1)),
            description='Neighbor Order:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='250px')
        )
        self.widgets["neighbor_order_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        container = VBox([
            HBox([
                self.widgets["neighbor_order"]
            ]),
            self.widgets["neighbor_order_error"]
        ])
        container.add_class('plugin-compact-container')
        self.widgets["config_container"].children = [container]
        self.widgets["config_container"].add_class('plugin-top-spacing')
        self.widgets["config_container"].add_class('plugin-bottom-spacing')
        self.widgets["neighbor_order"].observe(lambda change: self._validate_plugin_input('ContactPlugin', 'neighbor_order', change.new), names='value')

    # ChemotaxisPlugin widgets
    def create_chemotaxis_widgets(self, saved_values):
        """Widgets for ChemotaxisPlugin"""
        defaults = ChemotaxisPlugin().spec_dict
        self.widgets["field"] = widgets.Text(
            value=saved_values.get("field", defaults.get("field", "chemoattractant")),
            description='Field Name:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='250px')
        )
        self.widgets["field"].add_class('plugin-input-spacing')
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
        container.add_class('plugin-compact-container')
        self.widgets["config_container"].children = [container]
        self.widgets["config_container"].add_class('plugin-top-spacing')
        self.widgets["config_container"].add_class('plugin-bottom-spacing')
        self.widgets["field"].observe(lambda change: self._validate_plugin_input('ChemotaxisPlugin', 'field', change.new), names='value')
        self.widgets["lambda_val"].observe(lambda change: self._validate_plugin_input('ChemotaxisPlugin', 'lambda_val', change.new), names='value')

    # BoundaryPixelTrackerPlugin widgets
    def create_boundary_tracker_widgets(self, saved_values):
        """Widgets for BoundaryPixelTrackerPlugin"""
        # Get default values from class
        defaults = BoundaryPixelTrackerPlugin().spec_dict

        self.widgets["neighbor_order"] = widgets.BoundedIntText(
            value=saved_values.get("neighbor_order", defaults.get("neighbor_order", 2)),
            min=1, max=10,
            description='Neighbor Order:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='250px')
        )

        # Build UI with vertical spacing
        neighbor_box = VBox([
            self.widgets["neighbor_order"]
        ])
        # Remove extra spacing classes from neighbor_box
        self.widgets["config_container"].children = [neighbor_box]
        self.widgets["config_container"].add_class('plugin-top-spacing')
        self.widgets["config_container"].add_class('plugin-bottom-spacing')
        self.widgets["neighbor_order"].observe(lambda change: self._validate_plugin_input('BoundaryPixelTrackerPlugin', 'neighbor_order', change.new), names='value')

    def create_curvature_widgets(self, saved_values):
        """Widgets for CurvaturePlugin (no input boxes)"""
        self.widgets["config_container"].children = []
        self.widgets["config_container"].add_class('plugin-checkbox-bottom-spacing')
        self.widgets["config_container"].layout.display = 'block'

    def create_external_potential_widgets(self, saved_values):
        """Widgets for ExternalPotentialPlugin (no input boxes)"""
        self.widgets["config_container"].children = []
        self.widgets["config_container"].add_class('plugin-checkbox-bottom-spacing')
        self.widgets["config_container"].layout.display = 'block'

    def create_focal_point_plasticity_widgets(self, saved_values):
        """Widgets for FocalPointPlasticityPlugin (no input boxes)"""
        self.widgets["config_container"].children = []
        self.widgets["config_container"].add_class('plugin-checkbox-bottom-spacing')
        self.widgets["config_container"].layout.display = 'block'

    def create_ui(self):
        # For Cell Behavior plugins, show debug output at the bottom
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
        if not self.widgets["active"].value:
            return None

        config = {}

        # Handle plugins with rows (Volume, Surface)
        if "rows" in self.widgets:
            params = []
            if self.plugin_name == "VolumePlugin":
                # Always save all cell types with their current values
                for row in self.widgets["rows"]:
                    cell_type = row["cell_type"].value if hasattr(row["cell_type"], 'value') else row["cell_type"].description
                    target_volume = row["target_volume"].value if hasattr(row["target_volume"], 'value') else self.param_cache.get(cell_type, {}).get("target_volume", 25.0)
                    lambda_volume = row["lambda_volume"].value if hasattr(row["lambda_volume"], 'value') else self.param_cache.get(cell_type, {}).get("lambda_volume", 2.0)
                    params.append({
                        "CellType": cell_type,
                        "target_volume": target_volume,
                        "lambda_volume": lambda_volume
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

        # Add other plugin-specific values
        for key, widget in self.widgets.items():
            if key not in ["active", "config_container", "rows", "add_btn"]:
                config[key] = widget.value

        # Validate configuration using plugin class
        try:
            plugin_instance = self.plugin_class()
            for k, v in config.items():
                setattr(plugin_instance, k, v)
            # Pass context if available
            if self.parent_ui and hasattr(self.parent_ui, 'potts_core') and hasattr(self.parent_ui, 'cell_type_plugin'):
                plugin_instance.validate(self.parent_ui.potts_core, self.parent_ui.cell_type_plugin)
            else:
                plugin_instance.validate()
            return config
        except Exception as e:
            print(f"Validation error for {self.plugin_name}: {str(e)}")
            return None

    def reset(self):
        self.widgets["active"].value = False
        self.widgets["config_container"].layout.display = 'none'
        default = self.default_instance.spec_dict

        # Reset to plugin defaults
        if "rows" in self.widgets:
            self.widgets["rows"] = []
            if "params" in default:
                if self.plugin_name == "VolumePlugin":
                    self.create_volume_widgets(default)
                elif self.plugin_name == "SurfacePlugin":
                    self.create_surface_widgets(default)

        # Reset other widgets
        for key, widget in self.widgets.items():
            if key not in ["active", "config_container", "rows", "add_btn"] and key in default:
                widget.value = default[key]

    def _validate_plugin_input(self, plugin, field, value=None):
        with self.output:
            print(f"[DEBUG] _validate_plugin_input called for {plugin}.{field} with value={value}")
        # Map plugin to class
        plugin_class_map = {
            'AdhesionFlexPlugin': AdhesionFlexPlugin,
            'ChemotaxisPlugin': ChemotaxisPlugin,
            'ContactPlugin': ContactPlugin
        }
        widget_map = self.widgets
        error_widget_name = f"{field}_error"
        input_widget = widget_map.get(field)
        error_widget = widget_map.get(error_widget_name)
        if not input_widget or not error_widget:
            with self.output:
                print(f"[DEBUG] input_widget or error_widget missing for {field}")
            return
        # Always clear error and highlight before validation (Potts Core style)
        error_widget.value = ""
        error_widget.layout.display = 'none'
        if hasattr(input_widget, 'remove_class'):
            input_widget.remove_class('error-input')
        # Only backend validation
        try:
            plugin_instance = plugin_class_map[plugin]()
            # Use the latest value from the observer if provided
            if value is not None:
                setattr(plugin_instance, field, value)
            else:
                setattr(plugin_instance, field, input_widget.value)

            # Pass dependencies if available
            if self.parent_ui and hasattr(self.parent_ui, 'potts_core') and hasattr(self.parent_ui, 'cell_type_plugin'):
                plugin_instance.validate(self.parent_ui.potts_core, self.parent_ui.cell_type_plugin)
            else:
                plugin_instance.validate()
            with self.output:
                print(f"[DEBUG] Backend validation PASSED for {plugin}.{field} value={value if value is not None else input_widget.value}")
            # If validation passes, ensure error is cleared
            error_widget.value = ""
            error_widget.layout.display = 'none'
            if hasattr(input_widget, 'remove_class'):
                input_widget.remove_class('error-input')
        except Exception as e:
            with self.output:
                print(f"[DEBUG] Backend validation FAILED for {plugin}.{field} value={value if value is not None else input_widget.value}: {e}")
            # Show error, splitting multiple errors onto separate lines
            msg = str(e)
            if '\n' in msg:
                parts = [line.strip() for line in msg.split('\n') if line.strip()]
            else:
                parts = []
                for part in msg.split('Could not'):
                    part = part.strip()
                    if part:
                        if not part.startswith('Could not'):
                            part = 'Could not ' + part
                        parts.append(part)
                if not parts:
                    parts = [msg]
            html = '<br>'.join(f'⚠️ {p}' for p in parts)
            error_widget.value = f'<span style="color: red; font-size: 12px;">{html}</span>'
            error_widget.layout.display = 'block'
            if hasattr(input_widget, 'add_class'):
                input_widget.add_class('error-input')

    def update_cell_types(self, cell_types):
        # Called externally when cell types change
        self.cell_types = cell_types
        if self.plugin_name == "VolumePlugin":
            self.create_volume_widgets({"params": [self.param_cache.get(ct, {"CellType": ct, "target_volume": 25.0, "lambda_volume": 2.0}) for ct in cell_types]})

    def save_adhesionflex_values(self):
        # Get current values from the widgets
        neighbor_order = self.widgets['neighbor_order'].value
        max_distance = self.widgets['max_distance'].value
        # Update the config dict (assume self.parent_ui.current_config() returns the config dict)
        if self.parent_ui and hasattr(self.parent_ui, 'current_config'):
            config = self.parent_ui.current_config()
            if 'Plugins' not in config:
                config['Plugins'] = {}
            config['Plugins']['AdhesionFlexPlugin'] = {
                'neighbor_order': neighbor_order,
                'max_distance': max_distance
            }

    def remove_adhesionflex_values(self):
        if self.parent_ui and hasattr(self.parent_ui, 'current_config'):
            config = self.parent_ui.current_config()
            if 'Plugins' in config and 'AdhesionFlexPlugin' in config['Plugins']:
                del config['Plugins']['AdhesionFlexPlugin']

    def _on_adhesionflex_input_change(self, change):
        print("[DEBUG] AdhesionFlex input changed")
        if self.widgets["active"].value:  # Only save if plugin is enabled
            if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
                print("[DEBUG] Calling save_to_json from input change")
                self.parent_ui.save_to_json()


class PluginsTab:
    def __init__(self, saved_plugins, cell_types, parent_ui=None):
        self.widgets = {}
        self.plugin_widgets = {}
        self.cell_types = cell_types
        self.parent_ui = parent_ui  # <-- add this line
        self.create_widgets(saved_plugins or DEFAULTS["Plugins"])

    def create_widgets(self, saved_plugins):
        # Create tabs for plugin categories
        self.widgets["tabs"] = widgets.Tab()

        # Create categories
        behavior_plugins = []
        constraint_plugins = []
        tracker_plugins = []
        other_plugins = []

        # Plugin classes mapping
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
            plugin_widget = PluginWidget(plugin_name, plugin_class, plugin_values, self.cell_types, parent_ui=self.parent_ui)
            self.plugin_widgets[plugin_name] = plugin_widget

            # Categorize
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

        # Create tab content
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
        ).add_class('button-spacing')

    def get_config(self):
        config = {}
        for plugin_name, widget in self.plugin_widgets.items():
            plugin_config = widget.get_config()
            if plugin_config is not None:
                config[plugin_name] = plugin_config
        return config

    def reset(self):
        for widget in self.plugin_widgets.values():
            widget.reset()

    def update_cell_types(self, cell_types):
        self.cell_types = cell_types
        for widget in self.plugin_widgets.values():
            if "rows" in widget.widgets:
                for row in widget.widgets["rows"]:
                    if "cell_type" in row:
                        row["cell_type"].options = cell_types

    def create_ui(self):
        return VBox([
            self.widgets["tabs"],
            self.widgets["reset_button"]
        ])


class PottsWidget:
    def __init__(self, saved_values):
        self.widgets = {}
        self.defaults = PottsCore().spec_dict
        self.create_widgets(saved_values or self.defaults)

    def create_widgets(self, saved_values):
        # Dimension inputs
        self.widgets["dim_x"] = widgets.IntText(
            value=saved_values.get("dim_x", self.defaults["dim_x"]),
            min=1, description='X Dimension:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["dim_y"] = widgets.IntText(
            value=saved_values.get("dim_y", self.defaults["dim_y"]),
            min=1, description='Y Dimension:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["dim_z"] = widgets.IntText(
            value=saved_values.get("dim_z", self.defaults["dim_z"]),
            min=1, description='Z Dimension:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )

        # Error display for dim_y
        self.widgets["dim_x_error"] = HTML(
            value="",
            layout=Layout(
                margin='2px 0 5px 0',
                display='none'
            )
        )
        self.widgets["dim_y_error"] = HTML(
            value="",
            layout=Layout(
                margin='2px 0 5px 0',
                display='none'
            )
        )
        self.widgets["dim_z_error"] = HTML(
            value="",
            layout=Layout(
                margin='2px 0 5px 0',
                display='none'
            )
        )

        # Core parameters
        self.widgets["steps"] = widgets.IntText(
            value=saved_values.get("steps", self.defaults["steps"]),
            min=1, description='MC Steps:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["steps_error"] = HTML(
            value="",
            layout=Layout(
                margin='2px 0 5px 0',
                display='none'
            )
        )
        self.widgets["fluctuation_amplitude"] = widgets.FloatText(
            value=saved_values.get("fluctuation_amplitude", self.defaults["fluctuation_amplitude"]),
            min=0.0, description='Fluctuation Amplitude:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='250px')
        )
        self.widgets["fluctuation_amplitude_error"] = HTML(
            value="",
            layout=Layout(
                margin='2px 0 5px 0',
                display='none'
            )
        )
        self.widgets["neighbor_order"] = widgets.BoundedIntText(
            value=saved_values.get("neighbor_order", self.defaults["neighbor_order"]),
            min=1, max=20, description='Neighbor Order:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["neighbor_order_error"] = HTML(
            value="",
            layout=Layout(
                margin='2px 0 5px 0',
                display='none'
            )
        )
        self.widgets["lattice_type"] = widgets.Dropdown(
            options=['Cartesian', 'Hexagonal'],
            value=saved_values.get("lattice_type", self.defaults["lattice_type"]),
            description='Lattice Type:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )

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
        return {
            "dim_x": self.widgets["dim_x"].value,
            "dim_y": self.widgets["dim_y"].value,
            "dim_z": self.widgets["dim_z"].value,
            "steps": self.widgets["steps"].value,
            "fluctuation_amplitude": self.widgets["fluctuation_amplitude"].value,
            "boundary_x": self.widgets["boundary_x"].value,
            "boundary_y": self.widgets["boundary_y"].value,
            "boundary_z": self.widgets["boundary_z"].value,
            "neighbor_order": self.widgets["neighbor_order"].value,
            "lattice_type": self.widgets["lattice_type"].value
        }

    def reset(self):
        for key, widget in self.widgets.items():
            if key != "reset_button" and key in self.defaults:
                widget.value = self.defaults[key]

    def create_ui(self):
        # Create UI elements with CSS classes for spacing
        dim_x_box = VBox([self.widgets["dim_x"], self.widgets["dim_x_error"]])
        dim_x_box.add_class('hbox-item-spacing')

        dim_y_box = VBox([self.widgets["dim_y"], self.widgets["dim_y_error"]])
        dim_y_box.add_class('hbox-item-spacing')

        dim_z_box = VBox([self.widgets["dim_z"], self.widgets["dim_z_error"]])
        dim_z_box.add_class('vbox-no-margin')

        dimensions_row = HBox([dim_x_box, dim_y_box, dim_z_box])
        dimensions_row.add_class('vbox-row-spacing')

        steps_box = VBox([self.widgets["steps"], self.widgets["steps_error"]])
        steps_box.add_class('hbox-item-spacing')

        amplitude_box = VBox([self.widgets["fluctuation_amplitude"], self.widgets["fluctuation_amplitude_error"]])
        amplitude_box.add_class('vbox-no-margin')

        core_params_row = HBox([steps_box, amplitude_box])
        core_params_row.add_class('vbox-row-spacing')

        boundary_x_box = widgets.Box([self.widgets["boundary_x"]])
        boundary_x_box.add_class('hbox-item-spacing')

        boundary_y_box = widgets.Box([self.widgets["boundary_y"]])
        boundary_y_box.add_class('hbox-item-spacing')

        boundaries_row = HBox([boundary_x_box, boundary_y_box, self.widgets["boundary_z"]])
        boundaries_row.add_class('vbox-row-spacing')

        neighbor_order_box = VBox([self.widgets["neighbor_order"], self.widgets["neighbor_order_error"]])
        neighbor_order_box.add_class('hbox-item-spacing')

        advanced_row = HBox([neighbor_order_box, self.widgets["lattice_type"]])
        advanced_row.add_class('vbox-row-spacing')

        return VBox([
            HTML("<b>Potts Core Parameters<b>"),
            dimensions_row,
            core_params_row,
            HTML("<b>Boundary Conditions:</b>"),
            boundaries_row,
            HTML("<b>Advanced Settings:</b>"),
            advanced_row,
            self.widgets["reset_button"]
        ])


class CellTypeWidget:
    def __init__(self, saved_entries, on_change=None):
        self.on_change = on_change
        self.next_id = 0
        self.celltype_entries =[]

        entries = saved_entries or DEFAULTS["CellType"]
        for entry in entries:
            if isinstance(entry, dict):
                self.add_entry(entry["Cell type"], entry.get("freeze", False))
            else:
                self.add_entry(entry, False)

        if not any(entry["Cell type"] == "Medium" for entry in self.celltype_entries):
            self.add_entry("Medium", False)

        # self.celltype_entries = saved_entries or DEFAULTS["CellType"].copy()
        self.widgets = {}
        self.create_widgets()
        self.setup_event_handlers()
        self.update_celltype_display()

    def add_entry(self, name, freeze):
        self.celltype_entries.append({
            "Cell type": name,
            # "id": self.next_id,
            "freeze": freeze
        })
        # self.next_id += 1

    def create_widgets(self):
        # Display area for current cell types
        self.widgets["display_box"] = VBox(
            layout=Layout(padding='10px')
        )

        # Input widgets
        self.widgets["name"] = Text(
            placeholder="Cell type name",
            description="Name:",
            style={'description_width': 'initial'}
        )
        self.widgets["freeze"] = Checkbox(
            value=False,
            description="Freeze",
            indent=False
        )
        self.widgets["add_button"] = Button(
            description="Add Cell Type",
            button_style="success"
        )

        # Reset button
        self.widgets["reset_button"] = Button(
            description="Reset Cell Types",
            button_style='warning'
        )

    def setup_event_handlers(self):
        self.widgets["add_button"].on_click(self.on_add_clicked)
        self.widgets["reset_button"].on_click(self.reset)

    def on_add_clicked(self, _):
        name = self.widgets["name"].value.strip()

        if not name:
            self.widgets["name"].placeholder = "Please enter a name!"
            return

        # Check for duplicates
        if any(entry["Cell type"] == name for entry in self.celltype_entries):
            self.widgets["name"].value = ""
            self.widgets["name"].placeholder = f"{name} already exists!"
            return

        self.celltype_entries.append({
            "Cell type": name,
            "freeze": self.widgets["freeze"].value
        })

        self.update_celltype_display()
        self.widgets["name"].value = ""
        self.widgets["name"].placeholder = "Cell type name"

    def update_celltype_display(self):
        # Display cell types in a table format
        n = len(self.celltype_entries)
        if n == 0:
            self.widgets["display_box"].children = [HTML("<i>No cell types defined.</i>")]
            return

        # Table border styles
        row_border = '1px solid #e0e0e0'  # light grey
        header_border = '2px solid #bdbdbd'  # slightly bolder grey

        # Create table header with bottom border
        header = [
            HTML(f"<b style='display:block; padding:2px 8px;'>Cell Type</b>", layout=Layout(border=f'0 0 {header_border} 0')),
            HTML(f"<b style='display:block; padding:2px 8px;'>Frozen</b>", layout=Layout(border=f'0 0 {header_border} 0')),
            HTML(f"<b style='display:block; padding:2px 8px;'>Remove</b>", layout=Layout(border=f'0 0 {header_border} 0'))
        ]
        grid = GridspecLayout(n + 1, 3, grid_gap="0px")
        for j, h in enumerate(header):
            grid[0, j] = h

        for i, entry in enumerate(self.celltype_entries):
            border_style = f'0 0 {row_border} 0' if i < n - 1 else '0'  # no border on last row
            grid[i + 1, 0] = Label(str(entry['Cell type']), layout=Layout(border=border_style, padding='2px 8px'))
            grid[i + 1, 1] = Label("Yes" if entry.get('freeze', False) else "No", layout=Layout(border=border_style, padding='2px 8px'))
            remove_btn = Button(
                description="Remove",
                button_style='danger',
                layout=Layout(width='80px', border=border_style, padding='2px 8px')
            )
            def make_remove_handler(index):
                def handler(_):
                    del self.celltype_entries[index]
                    self.update_celltype_display()
                    if self.on_change:
                        self.on_change()
                return handler
            remove_btn.on_click(make_remove_handler(i))
            grid[i + 1, 2] = remove_btn

        self.widgets["display_box"].children = [grid]

    def get_config(self):
        return self.celltype_entries.copy()

    def get_cell_type_names(self):
        return [entry["Cell type"] for entry in self.celltype_entries]

    def reset(self, _=None):
        self.celltype_entries = []
        self.next_id = 0
        for entry in DEFAULTS["CellType"]:
            self.add_entry(entry["Cell type"], entry.get("freeze", False))
        self.update_celltype_display()

    def create_ui(self):
        # Apply spacing directly to widgets to avoid layout issues
        self.widgets["name"].add_class('small-right-spacing')
        self.widgets["freeze"].add_class('small-right-spacing')

        # Create input row with natural widget sizing
        input_row = HBox([
            self.widgets["name"],
            self.widgets["freeze"],
            self.widgets["add_button"]
        ])
        input_row.add_class('vbox-row-spacing')

        # Create reset button with spacing
        reset_button_box = VBox([self.widgets["reset_button"]])
        reset_button_box.add_class('vbox-row-spacing')

        return VBox([
            HTML("<b>Cell Types</b>"),
            self.widgets["display_box"],
            input_row,
            reset_button_box
        ])


class SpecificationSetupUI:
    def __init__(self):
        self.widgets = {}
        self.saved_values = self.load_saved_values()

        # Initialize core components with defaults
        self.metadata = Metadata()
        self.potts_core = PottsCore()
        self.cell_type_plugin = CellTypePlugin()

        # Apply saved values if available
        self.apply_saved_values()

        # Initialize widgets
        self.create_metadata_widgets()
        self.potts_widget = PottsWidget(self.saved_values.get("PottsCore"))
        # Use a custom callback for cell type changes
        self.celltype_widget = CellTypeWidget(
            self.saved_values.get("CellType"),
            on_change=self.cell_types_changed
        )
        self.plugins_tab = PluginsTab(
            self.saved_values.get("Plugins", {}),
            self.celltype_widget.get_cell_type_names(),
            parent_ui=self
        )

        # Create the UI
        self.create_ui()
        self.setup_event_handlers()

    def cell_types_changed(self):
        self.update_plugin_cell_types()
        self.save_to_json()

    def update_plugin_cell_types(self):
        cell_types = self.celltype_widget.get_config()
        cell_type_names = [entry["Cell type"] for entry in cell_types]
        # Update VolumePlugin
        volume_widget = self.plugins_tab.plugin_widgets.get("VolumePlugin")
        if volume_widget:
            volume_widget.update_cell_types(cell_type_names)

    def apply_saved_values(self):
        """Apply saved values to core components"""
        # Metadata
        if "Metadata" in self.saved_values:
            for key, value in self.saved_values["Metadata"].items():
                if hasattr(self.metadata, key):
                    setattr(self.metadata, key, value)

        # Potts Core
        if "PottsCore" in self.saved_values:
            for key, value in self.saved_values["PottsCore"].items():
                if hasattr(self.potts_core, key):
                    setattr(self.potts_core, key, value)

        # Cell Types
        if "CellType" in self.saved_values:
            for entry in self.saved_values["CellType"]:
                if isinstance(entry, dict):
                    cell_type_name = entry["Cell type"]
                    # Skip Medium as it's already in CellTypePlugin by default with ID 0
                    if cell_type_name == "Medium":
                        continue
                    self.cell_type_plugin.cell_type_append(
                        cell_type_name,
                        frozen=entry.get("freeze", False)
                    )
                else:
                    # Old format (string only) - skip Medium
                    if entry != "Medium":
                        self.cell_type_plugin.cell_type_append(entry)

    def create_metadata_widgets(self):
        """Metadata widgets"""
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
        self.widgets["reset_button"] = Button(
            description="Reset All to Defaults",
            button_style='danger'
        )

    def load_saved_values(self):
        try:
            if os.path.exists(SAVE_FILE):
                with open(SAVE_FILE, 'r') as f:
                    return json.load(f)
            return json.loads(json.dumps(DEFAULTS))
        except (json.JSONDecodeError, IOError):
            return json.loads(json.dumps(DEFAULTS))

    def current_config(self):
        return {
            "Metadata": self.metadata.spec_dict,
            "PottsCore": self.potts_core.spec_dict,
            "CellType": self.celltype_widget.get_config(),
            "Plugins": self.plugins_tab.get_config()
        }

    def save_to_json(self, _=None):
       config = self.current_config()
       print("Saving config:", config)
       if config:  # Only save if not empty
           with open(SAVE_FILE, 'w') as f:
               json.dump(config, f, indent=4)
       else:
           print("Warning: Attempted to save empty config!")

    def setup_event_handlers(self):
        # Metadata handlers
        self.widgets["num_processors"].observe(
            lambda change: self.update_metadata('num_processors', change.new),
            names='value'
        )
        self.widgets["debug_output_frequency"].observe(
            lambda change: self.update_metadata('debug_output_frequency', change.new),
            names='value'
        )

        # Connect reset handlers
        self.potts_widget.widgets["reset_button"].on_click(
            lambda _: self.reset_potts_tab()
        )
        self.celltype_widget.widgets["reset_button"].on_click(
            lambda _: self.reset_celltype_tab()
        )
        self.plugins_tab.widgets["reset_button"].on_click(
            lambda _: self.save_to_json()
            # lambda _: self.reset_plugins_tab()
        )
        self.widgets["reset_metadata_button"].on_click(
            lambda _: self.reset_metadata_tab()
        )

        # Global reset button
        self.widgets["reset_button"].on_click(
            lambda _: self.reset_all()
        )

        # PottsCore widget handlers
        for name, widget in self.potts_widget.widgets.items():
            if hasattr(widget, 'observe') and name != "reset_button":
                widget.observe(
                    lambda change, prop=name: self.update_potts_core(prop, change.new),
                    names='value'
                )

        # CellType handlers
        self.celltype_widget.widgets["add_button"].on_click(
            lambda _: self.update_cell_types()
        )

        # Auto-save on changes
        for widget in [self.widgets["num_processors"], self.widgets["debug_output_frequency"]]:
            widget.observe(lambda _: self.save_to_json(), names='value')

    def update_metadata(self, property_name, value):
        try:
            setattr(self.metadata, property_name, value)  # This may raise
            self.save_to_json()
            self.clear_metadata_error(property_name)
        except Exception as e:
            self.show_metadata_error(property_name, str(e))

    def show_metadata_error(self, property_name, message):
        """Display error message for a metadata property and highlight the input box"""
        error_widget_name = f"{property_name}_error"
        input_widget = self.widgets.get(property_name)
        if error_widget_name in self.widgets:
            error_widget = self.widgets[error_widget_name]
            error_widget.value = f'<span style="color: red; font-size: 12px;">⚠️ {message}</span>'
            error_widget.layout.display = 'block'
        if input_widget is not None and hasattr(input_widget, 'add_class'):
            input_widget.add_class('error-input')

    def clear_metadata_error(self, property_name):
        """Clear error message for a metadata property and remove input highlight"""
        error_widget_name = f"{property_name}_error"
        input_widget = self.widgets.get(property_name)
        if error_widget_name in self.widgets:
            error_widget = self.widgets[error_widget_name]
            error_widget.value = ""
            error_widget.layout.display = 'none'
        if input_widget is not None and hasattr(input_widget, 'remove_class'):
            input_widget.remove_class('error-input')

    def update_potts_core(self, property_name, value):
        try:
            setattr(self.potts_core, property_name, value)  # This may raise
            self.save_to_json()
            self.clear_potts_error(property_name)
        except Exception as e:
            self.show_potts_error(property_name, str(e))

    def show_potts_error(self, property_name, message):
        """Display error message for a potts property and highlight the input box"""
        error_widget_name = f"{property_name}_error"
        input_widget = self.potts_widget.widgets.get(property_name)
        if error_widget_name in self.potts_widget.widgets:
            error_widget = self.potts_widget.widgets[error_widget_name]
            # Wrap message in red HTML styling with icon
            error_widget.value = f'<span style="color: red; font-size: 12px;">⚠️ {message}</span>'
            error_widget.layout.display = 'block'
        if input_widget is not None and hasattr(input_widget, 'add_class'):
            input_widget.add_class('error-input')

    def clear_potts_error(self, property_name):
        """Clear error message for a potts property and remove input highlight"""
        error_widget_name = f"{property_name}_error"
        input_widget = self.potts_widget.widgets.get(property_name)
        if error_widget_name in self.potts_widget.widgets:
            error_widget = self.potts_widget.widgets[error_widget_name]
            error_widget.value = ""
            error_widget.layout.display = 'none'
        if input_widget is not None and hasattr(input_widget, 'remove_class'):
            input_widget.remove_class('error-input')

    def create_ui(self):
        tabs = Tab()
        tabs.children = [
            self.create_metadata_tab(),
            self.potts_widget.create_ui(),
            self.celltype_widget.create_ui(),
            self.plugins_tab.create_ui()
        ]
        tabs.set_title(0, 'Metadata')
        tabs.set_title(1, 'Potts Core')
        tabs.set_title(2, 'Cell Types')
        tabs.set_title(3, 'Plugins')

        # Create save button
        save_button = Button(
            description="Save Configuration",
            button_style='success'
        )
        save_button.on_click(self.save_to_json)

        # Create button row with spacing between buttons
        reset_button_box = VBox([self.widgets["reset_button"]])
        reset_button_box.add_class('hbox-item-spacing')

        save_button_box = VBox([save_button])
        save_button_box.add_class('vbox-no-margin')

        button_row = HBox([reset_button_box, save_button_box])
        button_row.add_class('vbox-row-spacing')

        display(VBox([
            tabs,
            button_row
        ]))

    def create_metadata_tab(self):
        # Create UI elements with CSS classes for spacing
        num_processors_box = VBox([self.widgets["num_processors"], self.widgets["num_processors_error"]])
        num_processors_box.add_class('vbox-row-spacing')

        debug_frequency_box = VBox([self.widgets["debug_output_frequency"], self.widgets["debug_output_frequency_error"]])
        debug_frequency_box.add_class('vbox-row-spacing')

        reset_button_box = VBox([self.widgets["reset_metadata_button"]])
        reset_button_box.add_class('vbox-row-spacing')

        return VBox([
            HTML("<b>Simulation Metadata</b>"),
            num_processors_box,
            debug_frequency_box,
            reset_button_box
        ])

    def reset_potts_tab(self):
        self.potts_widget.reset()
        for prop, value in PottsCore().spec_dict.items():
            if hasattr(self.potts_core, prop):
                setattr(self.potts_core, prop, value)
        self.save_to_json()

    def reset_celltype_tab(self):
        self.celltype_widget.reset()
        self.cell_type_plugin = CellTypePlugin()
        self.plugins_tab.update_cell_types(self.celltype_widget.get_cell_type_names())
        self.save_to_json()

    def reset_plugins_tab(self):
        self.plugins_tab.reset()
        self.save_to_json()

    def reset_metadata_tab(self):
        self.widgets["num_processors"].value = Metadata().num_processors
        self.widgets["debug_output_frequency"].value = Metadata().debug_output_frequency
        self.metadata = Metadata()
        self.save_to_json()

    def reset_all(self):
        self.reset_metadata_tab()
        self.reset_potts_tab()
        self.reset_celltype_tab()
        self.reset_plugins_tab()

    def update_cell_types(self):
        self.cell_type_plugin = CellTypePlugin()
        for entry in self.celltype_widget.celltype_entries:
            self.cell_type_plugin.cell_type_append(
                entry["Cell type"],
                # type_id=entry["id"],
                frozen=entry["freeze"]
            )
        self.plugins_tab.update_cell_types(self.celltype_widget.get_cell_type_names())
        self.save_to_json()
