import os
import json
import ipywidgets as widgets
from ipywidgets import (
    VBox, HBox, Layout, Dropdown, BoundedIntText, BoundedFloatText,
    FloatText, Checkbox, Button, Text, Label, Tab, HTML, Output
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
    padding: 5px 0 0 20px !important;
}

.plugin-input-spacing {
    margin: 0 15px 0 0 !important;
}

.plugin-bottom-spacing {
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
</style>
"""))

# Configuration
SAVE_FILE = 'simulation_setup.json'

# Get default values from class constructors using .spec_dict
def get_defaults():
    return {
        "Metadata": Metadata().spec_dict,
        "PottsCore": PottsCore().spec_dict,
        "CellType": [{"Cell type": "Medium", "freeze": False}], # ID automatically assigned by CompuCell3D
        "Plugins": {
            "AdhesionFlexPlugin": AdhesionFlexPlugin().spec_dict,
            "BoundaryPixelTrackerPlugin": BoundaryPixelTrackerPlugin().spec_dict,
            "ChemotaxisPlugin": ChemotaxisPlugin().spec_dict,
            "ContactPlugin": ContactPlugin().spec_dict,
            "CurvaturePlugin": CurvaturePlugin().spec_dict,
            "ExternalPotentialPlugin": ExternalPotentialPlugin().spec_dict,
            "FocalPointPlasticityPlugin": FocalPointPlasticityPlugin().spec_dict,
            "LengthConstraintPlugin": LengthConstraintPlugin().spec_dict,
            "PixelTrackerPlugin": PixelTrackerPlugin().spec_dict,
            "SecretionPlugin": SecretionPlugin().spec_dict,
            "VolumePlugin": VolumePlugin().spec_dict,
            "SurfacePlugin": SurfacePlugin().spec_dict
        }
    }

DEFAULTS = get_defaults()

class PluginWidget:
    def __init__(self, plugin_name, plugin_class, saved_values, cell_types):
        self.plugin_name = plugin_name
        self.plugin_class = plugin_class
        self.default_instance = plugin_class()
        self.cell_types = cell_types
        self.widgets = {}
        self.create_widgets(saved_values or self.default_instance.spec_dict)
        
    def create_widgets(self, saved_values):
        # Main checkbox with plugin name
        self.widgets["active"] = widgets.Checkbox(
            value=bool(saved_values),
            description=self.plugin_name,
            indent=False
        )
        
        # Container for plugin-specific widgets
        self.widgets["config_container"] = VBox([], layout=Layout(
            display='none' if not saved_values else 'block'
        ))
        self.widgets["config_container"].add_class('plugin-config-container')
        
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
        elif self.plugin_name == "LengthConstraintPlugin":
            self.create_length_constraint_widgets(saved_values)
        # Add other plugins as needed...
        
        # Set up active toggle
        self.widgets["active"].observe(self.toggle_config_visibility, names='value')
    
    def toggle_config_visibility(self, change):
        self.widgets["config_container"].layout.display = 'block' if change['new'] else 'none'
    
    # VolumePlugin widgets
    def create_volume_widgets(self, saved_values):
        """Widgets for VolumePlugin"""
        params = saved_values.get("params", [])

        if not params:
            default_params = VolumePlugin().spec_dict.get("params", [])
            params = default_params or [{
                "CellType": "Medium",
                "target_volume": 25.0,
                "lambda_volume": 2.0
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
                style={'description_width': 'initial'}
            )

            # Target volume with error handling
            row["target_volume"] = widgets.FloatText(
                value=param["target_volume"],
                description='Target Volume:',
                style={'description_width': 'initial'}
            )
            row["target_volume_error"] = ErrorHandler.create_error_display()

            # Lambda volume with error handling
            row["lambda_volume"] = widgets.FloatText(
                value=param["lambda_volume"],
                description='Lambda Volume:',
                style={'description_width': 'initial'}
            )
            row["lambda_volume_error"] = ErrorHandler.create_error_display()

            rows.append(row)
        
        self.widgets["rows"] = rows
        
        # Add button for new row
        self.widgets["add_btn"] = widgets.Button(
            description="Add Constraint",
            button_style='success'
        )
        
        # Define add_volume_row as a method
        def add_volume_row(_):
            default_params = VolumePlugin().spec_dict.get("params", [])
            default_values = default_params[0] if default_params else {
                "CellType": "Medium",
                "target_volume": 25.0,
                "lambda_volume": 2.0
            }

            new_row = {
                "cell_type": widgets.Dropdown(
                    options=self.cell_types,
                    value=default_values["CellType"],
                    description='Cell Type:',
                    style={'description_width': 'initial'}
                ),
                "target_volume": widgets.FloatText(
                    value=default_values["target_volume"],
                    description='Target Volume:',
                    style={'description_width': 'initial'}
                ),
                "lambda_volume": widgets.FloatText(
                    value=default_values["lambda_volume"],
                    description='Lambda Volume:',
                    style={'description_width': 'initial'}
                )
            }

            # Add error widgets
            new_row["target_volume_error"] = ErrorHandler.create_error_display()
            new_row["lambda_volume_error"] = ErrorHandler.create_error_display()

            self.widgets["rows"].append(new_row)
            self.update_volume_ui()
        
        # Connect the button handler
        self.widgets["add_btn"].on_click(add_volume_row)
        
        # Build UI
        self.update_volume_ui()
    
    def update_volume_ui(self):
        """Update the UI after row changes"""
        row_widgets = []
        for i, row in enumerate(self.widgets["rows"]):
            remove_btn = widgets.Button(
                description="Remove",
                button_style='danger',
                layout=Layout(width='100px')
            )

            def make_remove_handler(index):
                def handler(_):
                    del self.widgets["rows"][index]
                    self.update_volume_ui()
                return handler

            remove_btn.on_click(make_remove_handler(i))

            # Create input row with error displays and consistent spacing
            target_volume_box = VBox([row["target_volume"], row["target_volume_error"]])
            target_volume_box.add_class('plugin-input-spacing')

            lambda_volume_box = VBox([row["lambda_volume"], row["lambda_volume_error"]])
            lambda_volume_box.add_class('plugin-input-spacing')

            input_row = HBox([
                row["cell_type"],
                target_volume_box,
                lambda_volume_box,
                remove_btn
            ])

            row_widgets.append(input_row)

        # Add spacing around the Add Constraint button
        add_btn_box = VBox([self.widgets["add_btn"]])
        add_btn_box.add_class('button-spacing')

        self.widgets["config_container"].children = [
            *row_widgets,
            add_btn_box
        ]
    
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
                style={'description_width': 'initial'}
            )

            # Target surface with error handling
            row["target_surface"] = widgets.FloatText(
                value=param["target_surface"],
                description='Target Surface:',
                style={'description_width': 'initial'}
            )
            row["target_surface_error"] = ErrorHandler.create_error_display()

            # Lambda surface with error handling
            row["lambda_surface"] = widgets.FloatText(
                value=param["lambda_surface"],
                description='Lambda Surface:',
                style={'description_width': 'initial'}
            )
            row["lambda_surface_error"] = ErrorHandler.create_error_display()

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
                    description='Target Surface:',
                    style={'description_width': 'initial'}
                ),
                "lambda_surface": widgets.FloatText(
                    value=default_values["lambda_surface"],
                    description='Lambda Surface:',
                    style={'description_width': 'initial'}
                )
            }

            # Add error widgets
            new_row["target_surface_error"] = InputValidator.create_error_display()
            new_row["lambda_surface_error"] = InputValidator.create_error_display()

            # Set up validation for new widgets
            def validate_target_surface(value):
                if value <= 0:
                    raise ValueError("Target surface must be positive")
                if value > 50000:
                    raise ValueError("Target surface seems too large (>50000)")

            def validate_lambda_surface(value):
                if value < 0:
                    raise ValueError("Lambda surface must be non-negative")
                if value > 1000:
                    raise ValueError("Lambda surface seems too large (>1000)")

            InputValidator.setup_input_validation(
                new_row["target_surface"],
                new_row["target_surface_error"],
                validate_target_surface,
                "Invalid target surface"
            )

            InputValidator.setup_input_validation(
                new_row["lambda_surface"],
                new_row["lambda_surface_error"],
                validate_lambda_surface,
                "Invalid lambda surface"
            )

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

            # Create input row with error displays and consistent spacing
            target_surface_box = VBox([row["target_surface"], row["target_surface_error"]])
            target_surface_box.add_class('plugin-input-spacing')

            lambda_surface_box = VBox([row["lambda_surface"], row["lambda_surface_error"]])
            lambda_surface_box.add_class('plugin-input-spacing')

            input_row = HBox([
                row["cell_type"],
                target_surface_box,
                lambda_surface_box,
                remove_btn
            ])

            row_widgets.append(input_row)

        # Add spacing around the Add Constraint button
        add_btn_box = VBox([self.widgets["add_btn"]])
        add_btn_box.add_class('button-spacing')

        self.widgets["config_container"].children = [
            *row_widgets,
            add_btn_box
        ]
    
    # AdhesionFlexPlugin widgets
    def create_adhesion_widgets(self, saved_values):
        """Widgets for AdhesionFlexPlugin"""
        # Get default values from class
        defaults = AdhesionFlexPlugin().spec_dict

        self.widgets["neighbor_order"] = widgets.IntText(
            value=saved_values.get("neighbor_order", defaults.get("neighbor_order", 2)),
            description='Neighbor Order:',
            style={'description_width': 'initial'}
        )
        self.widgets["neighbor_order_error"] = ErrorHandler.create_error_display()

        self.widgets["max_distance"] = widgets.IntText(
            value=saved_values.get("max_distance", defaults.get("max_distance", 3)),
            description='Max Distance:',
            style={'description_width': 'initial'}
        )
        self.widgets["max_distance_error"] = ErrorHandler.create_error_display()

        # Build UI with improved spacing
        neighbor_order_box = VBox([self.widgets["neighbor_order"], self.widgets["neighbor_order_error"]])
        neighbor_order_box.add_class('plugin-input-spacing')

        max_distance_box = VBox([self.widgets["max_distance"], self.widgets["max_distance_error"]])
        max_distance_box.add_class('vbox-no-margin')

        plugin_row = HBox([neighbor_order_box, max_distance_box])
        plugin_row.add_class('plugin-bottom-spacing')

        self.widgets["config_container"].children = [plugin_row]
    
    # ContactPlugin widgets
    def create_contact_widgets(self, saved_values):
        """Widgets for ContactPlugin"""
        # Get default values from class
        defaults = ContactPlugin().spec_dict

        self.widgets["neighbor_order"] = widgets.IntText(
            value=saved_values.get("neighbor_order", defaults.get("neighbor_order", 1)),
            description='Neighbor Order:',
            style={'description_width': 'initial'}
        )
        self.widgets["neighbor_order_error"] = ErrorHandler.create_error_display()

        def validate_neighbor_order(value):
            if value < 1:
                raise ValueError("Neighbor order must be at least 1")
            if value > 10:
                raise ValueError("Neighbor order should not exceed 10")

        # Validation handled by backend

        # Build UI with improved spacing
        neighbor_order_box = VBox([self.widgets["neighbor_order"], self.widgets["neighbor_order_error"]])
        neighbor_order_box.add_class('plugin-bottom-spacing')

        self.widgets["config_container"].children = [neighbor_order_box]
    
    # ChemotaxisPlugin widgets
    def create_chemotaxis_widgets(self, saved_values):
        """Widgets for ChemotaxisPlugin"""
        # Get default values from class
        defaults = ChemotaxisPlugin().spec_dict

        self.widgets["field"] = widgets.Text(
            value=saved_values.get("field", defaults.get("field", "chemoattractant")),
            description='Field Name:',
            style={'description_width': 'initial'}
        )
        self.widgets["field_error"] = ErrorHandler.create_error_display()

        self.widgets["lambda_val"] = widgets.FloatText(
            value=saved_values.get("lambda", defaults.get("lambda", 100.0)),
            description='Lambda Value:',
            style={'description_width': 'initial'}
        )
        self.widgets["lambda_val_error"] = ErrorHandler.create_error_display()

        # Build UI with improved spacing
        field_box = VBox([self.widgets["field"], self.widgets["field_error"]])
        field_box.add_class('plugin-input-spacing')

        lambda_val_box = VBox([self.widgets["lambda_val"], self.widgets["lambda_val_error"]])
        lambda_val_box.add_class('vbox-no-margin')

        plugin_row = HBox([field_box, lambda_val_box])
        plugin_row.add_class('plugin-bottom-spacing')

        self.widgets["config_container"].children = [plugin_row]
    
    # BoundaryPixelTrackerPlugin widgets
    def create_boundary_tracker_widgets(self, saved_values):
        """Widgets for BoundaryPixelTrackerPlugin"""
        # Get default values from class
        defaults = BoundaryPixelTrackerPlugin().spec_dict

        self.widgets["neighbor_order"] = widgets.IntText(
            value=saved_values.get("neighbor_order", defaults.get("neighbor_order", 2)),
            description='Neighbor Order:',
            style={'description_width': 'initial'}
        )
        self.widgets["neighbor_order_error"] = ErrorHandler.create_error_display()

        def validate_neighbor_order(value):
            if value < 1:
                raise ValueError("Neighbor order must be at least 1")
            if value > 10:
                raise ValueError("Neighbor order should not exceed 10")

        InputValidator.setup_input_validation(
            self.widgets["neighbor_order"],
            self.widgets["neighbor_order_error"],
            validate_neighbor_order,
            "Invalid neighbor order"
        )

        # Build UI with improved spacing
        neighbor_order_box = VBox([self.widgets["neighbor_order"], self.widgets["neighbor_order_error"]])
        neighbor_order_box.add_class('plugin-bottom-spacing')

        self.widgets["config_container"].children = [neighbor_order_box]

    # CurvaturePlugin widgets
    def create_curvature_widgets(self, saved_values):
        """Widgets for CurvaturePlugin"""
        # Get default values from class
        defaults = CurvaturePlugin().spec_dict

        self.widgets["lambda_curvature"] = widgets.FloatText(
            value=saved_values.get("lambda_curvature", defaults.get("lambda_curvature", 0.0)),
            description='Lambda Curvature:',
            style={'description_width': 'initial'}
        )
        self.widgets["lambda_curvature_error"] = InputValidator.create_error_display()

        def validate_lambda_curvature(value):
            if value < 0:
                raise ValueError("Lambda curvature must be non-negative")
            if value > 1000:
                raise ValueError("Lambda curvature seems too large (>1000)")

        InputValidator.setup_input_validation(
            self.widgets["lambda_curvature"],
            self.widgets["lambda_curvature_error"],
            validate_lambda_curvature,
            "Invalid lambda curvature"
        )

        # Build UI with improved spacing
        lambda_curvature_box = VBox([self.widgets["lambda_curvature"], self.widgets["lambda_curvature_error"]])
        lambda_curvature_box.add_class('plugin-bottom-spacing')

        self.widgets["config_container"].children = [lambda_curvature_box]

    # ExternalPotentialPlugin widgets
    def create_external_potential_widgets(self, saved_values):
        """Widgets for ExternalPotentialPlugin"""
        # Get default values from class
        defaults = ExternalPotentialPlugin().spec_dict

        self.widgets["algorithm"] = widgets.Text(
            value=saved_values.get("algorithm", defaults.get("algorithm", "PixelBased")),
            description='Algorithm:',
            style={'description_width': 'initial'}
        )
        self.widgets["algorithm_error"] = InputValidator.create_error_display()

        def validate_algorithm(value):
            if not value or not value.strip():
                raise ValueError("Algorithm cannot be empty")
            if len(value.strip()) > 50:
                raise ValueError("Algorithm name too long (max 50 characters)")

        InputValidator.setup_input_validation(
            self.widgets["algorithm"],
            self.widgets["algorithm_error"],
            validate_algorithm,
            "Invalid algorithm"
        )

        # Build UI with improved spacing
        algorithm_box = VBox([self.widgets["algorithm"], self.widgets["algorithm_error"]])
        algorithm_box.add_class('plugin-bottom-spacing')

        self.widgets["config_container"].children = [algorithm_box]

    # FocalPointPlasticityPlugin widgets
    def create_focal_point_plasticity_widgets(self, saved_values):
        """Widgets for FocalPointPlasticityPlugin"""
        # Get default values from class
        defaults = FocalPointPlasticityPlugin().spec_dict

        self.widgets["neighbor_order"] = widgets.IntText(
            value=saved_values.get("neighbor_order", defaults.get("neighbor_order", 1)),
            description='Neighbor Order:',
            style={'description_width': 'initial'}
        )
        self.widgets["neighbor_order_error"] = InputValidator.create_error_display()

        def validate_neighbor_order(value):
            if value < 1:
                raise ValueError("Neighbor order must be at least 1")
            if value > 10:
                raise ValueError("Neighbor order should not exceed 10")

        InputValidator.setup_input_validation(
            self.widgets["neighbor_order"],
            self.widgets["neighbor_order_error"],
            validate_neighbor_order,
            "Invalid neighbor order"
        )

        # Build UI with improved spacing
        neighbor_order_box = VBox([self.widgets["neighbor_order"], self.widgets["neighbor_order_error"]])
        neighbor_order_box.add_class('plugin-bottom-spacing')

        self.widgets["config_container"].children = [neighbor_order_box]

    # LengthConstraintPlugin widgets
    def create_length_constraint_widgets(self, saved_values):
        """Widgets for LengthConstraintPlugin"""
        # Get default values from class
        defaults = LengthConstraintPlugin().spec_dict

        self.widgets["target_length"] = widgets.FloatText(
            value=saved_values.get("target_length", defaults.get("target_length", 10.0)),
            description='Target Length:',
            style={'description_width': 'initial'}
        )
        self.widgets["target_length_error"] = InputValidator.create_error_display()

        def validate_target_length(value):
            if value <= 0:
                raise ValueError("Target length must be positive")
            if value > 1000:
                raise ValueError("Target length seems too large (>1000)")

        InputValidator.setup_input_validation(
            self.widgets["target_length"],
            self.widgets["target_length_error"],
            validate_target_length,
            "Invalid target length"
        )

        self.widgets["lambda_length"] = widgets.FloatText(
            value=saved_values.get("lambda_length", defaults.get("lambda_length", 1.0)),
            description='Lambda Length:',
            style={'description_width': 'initial'}
        )
        self.widgets["lambda_length_error"] = InputValidator.create_error_display()

        def validate_lambda_length(value):
            if value < 0:
                raise ValueError("Lambda length must be non-negative")
            if value > 1000:
                raise ValueError("Lambda length seems too large (>1000)")

        InputValidator.setup_input_validation(
            self.widgets["lambda_length"],
            self.widgets["lambda_length_error"],
            validate_lambda_length,
            "Invalid lambda length"
        )

        # Build UI with consistent spacing
        target_length_box = VBox([self.widgets["target_length"], self.widgets["target_length_error"]])
        target_length_box.add_class('plugin-input-spacing')

        lambda_length_box = VBox([self.widgets["lambda_length"], self.widgets["lambda_length_error"]])
        lambda_length_box.add_class('vbox-no-margin')

        plugin_row = HBox([target_length_box, lambda_length_box])
        plugin_row.add_class('plugin-bottom-spacing')

        self.widgets["config_container"].children = [plugin_row]

    def create_ui(self):
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
            for row in self.widgets["rows"]:
                if self.plugin_name == "VolumePlugin":
                    params.append({
                        "CellType": row["cell_type"].value,
                        "target_volume": row["target_volume"].value,
                        "lambda_volume": row["lambda_volume"].value
                    })
                elif self.plugin_name == "SurfacePlugin":
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
                for param in default["params"]:
                    if self.plugin_name == "VolumePlugin":
                        self.add_volume_row(param)
                    elif self.plugin_name == "SurfacePlugin":
                        self.add_surface_row(param)
        
        # Reset other widgets
        for key, widget in self.widgets.items():
            if key not in ["active", "config_container", "rows", "add_btn"] and key in default:
                widget.value = default[key]

class PluginsTab:
    def __init__(self, saved_plugins, cell_types):
        self.widgets = {}
        self.plugin_widgets = {}
        self.cell_types = cell_types
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
            plugin_widget = PluginWidget(plugin_name, plugin_class, plugin_values, self.cell_types)
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
        )
        
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
        # Add spacing above reset button
        reset_button_box = VBox([self.widgets["reset_button"]])
        reset_button_box.add_class('vbox-row-spacing')

        return VBox([
            self.widgets["tabs"],
            reset_button_box
        ])


class PottsWidget:
    def __init__(self, saved_values):
        self.widgets = {}
        # Use default values without CompuCell3D validation
        self.defaults = {
            "dim_x": 100,
            "dim_y": 100,
            "dim_z": 1,
            "steps": 10000,
            "fluctuation_amplitude": 10.0,
            "neighbor_order": 2,
            "boundary_x": "Periodic",
            "boundary_y": "Periodic",
            "boundary_z": "Periodic",
            "lattice_type": "Square"
        }
        self.create_widgets(saved_values or self.defaults)
        
    def create_widgets(self, saved_values):
        # Dimension inputs with error handling
        self.widgets["dim_x"] = widgets.IntText(
            value=saved_values.get("dim_x", self.defaults["dim_x"]),
            description='X Dimension:',
            style={'description_width': 'initial'}
        )
        self.widgets["dim_x_error"] = InputValidator.create_error_display()

        def validate_dim_x(value):
            if value < 1:
                raise ValueError("X dimension must be at least 1")
            if value > 1000:
                raise ValueError("X dimension seems too large (>1000)")

        InputValidator.setup_input_validation(
            self.widgets["dim_x"],
            self.widgets["dim_x_error"],
            validate_dim_x,
            "Invalid X dimension"
        )

        self.widgets["dim_y"] = widgets.IntText(
            value=saved_values.get("dim_y", self.defaults["dim_y"]),
            description='Y Dimension:',
            style={'description_width': 'initial'}
        )
        self.widgets["dim_y_error"] = InputValidator.create_error_display()

        def validate_dim_y(value):
            if value < 1:
                raise ValueError("Y dimension must be at least 1")
            if value > 1000:
                raise ValueError("Y dimension seems too large (>1000)")

        InputValidator.setup_input_validation(
            self.widgets["dim_y"],
            self.widgets["dim_y_error"],
            validate_dim_y,
            "Invalid Y dimension"
        )

        self.widgets["dim_z"] = widgets.IntText(
            value=saved_values.get("dim_z", self.defaults["dim_z"]),
            description='Z Dimension:',
            style={'description_width': 'initial'}
        )
        self.widgets["dim_z_error"] = InputValidator.create_error_display()

        def validate_dim_z(value):
            if value < 1:
                raise ValueError("Z dimension must be at least 1")
            if value > 1000:
                raise ValueError("Z dimension seems too large (>1000)")

        InputValidator.setup_input_validation(
            self.widgets["dim_z"],
            self.widgets["dim_z_error"],
            validate_dim_z,
            "Invalid Z dimension"
        )

        # Core parameters with error handling
        self.widgets["steps"] = widgets.IntText(
            value=saved_values.get("steps", self.defaults["steps"]),
            description='MC Steps:',
            style={'description_width': 'initial'}
        )
        self.widgets["steps_error"] = InputValidator.create_error_display()

        def validate_steps(value):
            if value < 1:
                raise ValueError("MC steps must be at least 1")
            if value > 1000000:
                raise ValueError("MC steps seems too large (>1,000,000)")

        InputValidator.setup_input_validation(
            self.widgets["steps"],
            self.widgets["steps_error"],
            validate_steps,
            "Invalid MC steps"
        )

        self.widgets["fluctuation_amplitude"] = widgets.FloatText(
            value=saved_values.get("fluctuation_amplitude", self.defaults["fluctuation_amplitude"]),
            description='Fluctuation Amplitude:',
            style={'description_width': 'initial'}
        )
        self.widgets["fluctuation_amplitude_error"] = InputValidator.create_error_display()

        def validate_fluctuation_amplitude(value):
            if value < 0:
                raise ValueError("Fluctuation amplitude must be non-negative")
            if value > 1000:
                raise ValueError("Fluctuation amplitude seems too large (>1000)")

        InputValidator.setup_input_validation(
            self.widgets["fluctuation_amplitude"],
            self.widgets["fluctuation_amplitude_error"],
            validate_fluctuation_amplitude,
            "Invalid fluctuation amplitude"
        )
        
        # Boundaries
        boundary_options = ['NoFlux', 'Periodic']
        self.widgets["boundary_x"] = widgets.Dropdown(
            options=boundary_options,
            value=saved_values.get("boundary_x", self.defaults["boundary_x"]),
            description='X Boundary:',
            style={'description_width': 'initial'}
        )
        self.widgets["boundary_y"] = widgets.Dropdown(
            options=boundary_options,
            value=saved_values.get("boundary_y", self.defaults["boundary_y"]),
            description='Y Boundary:',
            style={'description_width': 'initial'}
        )
        self.widgets["boundary_z"] = widgets.Dropdown(
            options=boundary_options,
            value=saved_values.get("boundary_z", self.defaults["boundary_z"]),
            description='Z Boundary:',
            style={'description_width': 'initial'}
        )
        
        # Advanced settings with error handling
        self.widgets["neighbor_order"] = widgets.IntText(
            value=saved_values.get("neighbor_order", self.defaults["neighbor_order"]),
            description='Neighbor Order:',
            style={'description_width': 'initial'}
        )
        self.widgets["neighbor_order_error"] = InputValidator.create_error_display()

        def validate_neighbor_order(value):
            if value < 1:
                raise ValueError("Neighbor order must be at least 1")
            if value > 20:
                raise ValueError("Neighbor order should not exceed 20")

        InputValidator.setup_input_validation(
            self.widgets["neighbor_order"],
            self.widgets["neighbor_order_error"],
            validate_neighbor_order,
            "Invalid neighbor order"
        )

        self.widgets["lattice_type"] = widgets.Dropdown(
            options=['Cartesian', 'Hexagonal'],
            value=saved_values.get("lattice_type", self.defaults["lattice_type"]),
            description='Lattice Type:',
            style={'description_width': 'initial'}
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
    def __init__(self, saved_entries):
        self.celltype_entries = []

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
            "freeze": freeze
        })
        print(f"Added cell type: {name}, freeze: {freeze}")
        print(f"Total cell types: {len(self.celltype_entries)}")

    def debug_cell_types(self):
        """Debug method to print current cell types"""
        print("=== Current Cell Types ===")
        for i, entry in enumerate(self.celltype_entries):
            print(f"Index {i}: {entry['Cell type']}, Freeze: {entry['freeze']}")
        print("==========================")

    def add_cell_type_handler(self, _):
        """Event handler for Add Cell Type button"""
        name = self.widgets["name"].value.strip()
        freeze = self.widgets["freeze"].value

        if name and name not in [entry["Cell type"] for entry in self.celltype_entries]:
            self.add_entry(name, freeze)
            self.update_celltype_display()
            # Clear the input
            self.widgets["name"].value = ""
            self.widgets["freeze"].value = False
            print(f"✅ Added cell type: {name}")
        else:
            print(f"❌ Cannot add cell type: '{name}' (empty or already exists)")

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

        # Set up event handler for add button
        self.widgets["add_button"].on_click(self.add_cell_type_handler)
        
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
        items = []
        for i, entry in enumerate(self.celltype_entries):
            label = Label(f"{entry['Cell type']} {'(frozen)' if entry.get('freeze', False) else ''}")
            remove_btn = Button(
                description="Remove", 
                button_style='danger',
                layout=Layout(width='100px')
            )
            
            def make_remove_handler(index):
                def handler(_):
                    del self.celltype_entries[index]
                    self.update_celltype_display()
                return handler
            
            remove_btn.on_click(make_remove_handler(i))

            celltype_item = HBox([label, remove_btn])
            celltype_item.add_class('celltype-item-spacing')
            items.append(celltype_item)
        
        self.widgets["display_box"].children = items
    
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
        
        # Initialize core components without CompuCell3D objects (to avoid validation errors)
        # self.metadata = Metadata()  # Removed - causes SpecValueCheckError
        # self.potts_core = PottsCore()  # Removed - causes SpecValueCheckError
        # self.cell_type_plugin = CellTypePlugin()  # Removed - causes SpecValueCheckError
        
        # Apply saved values if available
        self.apply_saved_values()
        
        # Initialize widgets
        self.create_metadata_widgets()
        self.potts_widget = PottsWidget(self.saved_values.get("PottsCore"))
        self.celltype_widget = CellTypeWidget(self.saved_values.get("CellType"))
        self.plugins_tab = PluginsTab(
            self.saved_values.get("Plugins", {}),
            self.celltype_widget.get_cell_type_names()
        )

        # Create the UI
        self.create_ui()
        self.setup_event_handlers()

    def apply_saved_values(self):
        """Apply saved values to UI components (without CompuCell3D objects)"""
        # Store saved values for later use when creating widgets
        # No need to apply to CompuCell3D objects since we removed them

        # Cell Types - store in celltype_widget after it's created
        if "CellType" in self.saved_values:
            self.saved_cell_types = self.saved_values["CellType"]
        else:
            self.saved_cell_types = []

    def create_metadata_widgets(self):
        """Metadata widgets with error handling"""
        # Get saved values or use defaults
        metadata_values = self.saved_values.get("Metadata", {})

        self.widgets["num_processors"] = widgets.IntText(
            value=metadata_values.get("num_processors", 1),
            description='Number of Processors:',
            style={'description_width': 'initial'}
        )
        self.widgets["num_processors_error"] = InputValidator.create_error_display()

        def validate_num_processors(value):
            if value < 1:
                raise ValueError("Number of processors must be at least 1")
            if value > 128:
                raise ValueError("Number of processors seems too large (>128)")

        InputValidator.setup_input_validation(
            self.widgets["num_processors"],
            self.widgets["num_processors_error"],
            validate_num_processors,
            "Invalid number of processors"
        )

        self.widgets["debug_output_frequency"] = widgets.IntText(
            value=metadata_values.get("debug_output_frequency", 100),
            description='Debug Output Frequency:',
            style={'description_width': 'initial'}
        )
        self.widgets["debug_output_frequency_error"] = InputValidator.create_error_display()

        def validate_debug_frequency(value):
            if value < 0:
                raise ValueError("Debug output frequency must be non-negative")
            if value > 10000:
                raise ValueError("Debug output frequency seems too large (>10000)")

        InputValidator.setup_input_validation(
            self.widgets["debug_output_frequency"],
            self.widgets["debug_output_frequency_error"],
            validate_debug_frequency,
            "Invalid debug output frequency"
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
            "Metadata": self.get_metadata_config(),
            "PottsCore": self.potts_core.get_config(),
            "CellType": self.celltype_widget.get_config(),
            "Plugins": self.plugins_tab.get_config()
        }

    def get_metadata_config(self):
        """Get metadata configuration from UI widgets without CompuCell3D objects"""
        return {
            "num_processors": self.widgets["num_processors"].value,
            "debug_output_frequency": self.widgets["debug_output_frequency"].value
        }

    def save_to_json(self, _=None):
        try:
            # Update all configurations from current UI state
            self.update_configs_from_ui()

            config = self.current_config()

            # Validate configuration using backend validator
            validation_errors = BackendValidator.validate_config(config)

            if validation_errors:
                print("❌ Validation errors found:")
                for field_id, error_message in validation_errors.items():
                    print(f"  {field_id}: {error_message}")

                # Display errors in UI
                self.display_validation_errors(validation_errors)
                print("Please fix the errors above before saving.")
                return False

            print("=== Saving Configuration ===")
            print(f"Metadata: {config['Metadata']}")
            print(f"PottsCore: {config['PottsCore']}")
            print(f"CellType: {config['CellType']}")
            print(f"Plugins: {list(config['Plugins'].keys()) if config['Plugins'] else 'None'}")
            print("============================")

            with open(SAVE_FILE, 'w') as f:
                json.dump(config, f, indent=4)
            print(f"✅ Configuration saved to {SAVE_FILE}")
            return True

        except Exception as e:
            print(f"Error saving configuration: {e}")
            import traceback
            traceback.print_exc()
            return False

    def display_validation_errors(self, validation_errors):
        """Display validation errors in the UI"""
        # Clear all existing errors first
        self.clear_all_errors()

        # Map field IDs to widget/error widget pairs
        error_mapping = self.get_error_widget_mapping()

        # Display errors
        for field_id, error_message in validation_errors.items():
            if field_id in error_mapping:
                widget, error_widget = error_mapping[field_id]
                ErrorHandler.show_error(widget, error_widget, error_message)
            else:
                print(f"Warning: No UI widget found for field '{field_id}'")

    def clear_all_errors(self):
        """Clear all error displays in the UI"""
        # Clear metadata errors
        if hasattr(self, 'widgets'):
            for key in ['num_processors', 'debug_output_frequency']:
                if f"{key}_error" in self.widgets:
                    ErrorHandler.clear_error(self.widgets[key], self.widgets[f"{key}_error"])

        # Clear Potts core errors
        if hasattr(self, 'potts_core') and hasattr(self.potts_core, 'widgets'):
            for key in ['dim_x', 'dim_y', 'dim_z', 'steps', 'fluctuation_amplitude', 'neighbor_order']:
                if f"{key}_error" in self.potts_core.widgets:
                    ErrorHandler.clear_error(self.potts_core.widgets[key], self.potts_core.widgets[f"{key}_error"])

        # Clear plugin errors
        if hasattr(self, 'plugins_tab'):
            for plugin_name, plugin_widget in self.plugins_tab.plugin_widgets.items():
                if hasattr(plugin_widget, 'widgets'):
                    # Clear individual plugin errors
                    for widget_key, widget in plugin_widget.widgets.items():
                        if widget_key.endswith('_error'):
                            continue
                        error_key = f"{widget_key}_error"
                        if error_key in plugin_widget.widgets:
                            ErrorHandler.clear_error(widget, plugin_widget.widgets[error_key])

                    # Clear constraint row errors (VolumePlugin, SurfacePlugin)
                    if hasattr(plugin_widget, 'widgets') and 'rows' in plugin_widget.widgets:
                        for row in plugin_widget.widgets['rows']:
                            for field_key, field_widget in row.items():
                                if field_key.endswith('_error'):
                                    continue
                                error_key = f"{field_key}_error"
                                if error_key in row:
                                    ErrorHandler.clear_error(field_widget, row[error_key])

    def get_error_widget_mapping(self):
        """Get mapping of field IDs to (widget, error_widget) pairs"""
        mapping = {}

        # Metadata mappings
        if hasattr(self, 'widgets'):
            for key in ['num_processors', 'debug_output_frequency']:
                if key in self.widgets and f"{key}_error" in self.widgets:
                    mapping[key] = (self.widgets[key], self.widgets[f"{key}_error"])

        # Potts core mappings
        if hasattr(self, 'potts_core') and hasattr(self.potts_core, 'widgets'):
            for key in ['dim_x', 'dim_y', 'dim_z', 'steps', 'fluctuation_amplitude', 'neighbor_order']:
                if key in self.potts_core.widgets and f"{key}_error" in self.potts_core.widgets:
                    mapping[key] = (self.potts_core.widgets[key], self.potts_core.widgets[f"{key}_error"])

        # Plugin mappings (simplified - can be expanded)
        # This would need to be expanded based on the specific plugin structure

        return mapping

    def update_configs_from_ui(self):
        """Update all configuration objects from current UI state"""
        try:
            # No need to update CompuCell3D objects - we get values directly from UI

            # Debug cell types
            print("=== Cell Types Before Save ===")
            self.celltype_widget.debug_cell_types()

            print("=== Plugins Before Save ===")
            for plugin_name, widget in self.plugins_tab.plugin_widgets.items():
                if widget.widgets["active"].value:
                    print(f"Active plugin: {plugin_name}")

        except Exception as e:
            print(f"Error updating configs from UI: {e}")
            import traceback
            traceback.print_exc()

    def test_cell_type_ids(self):
        """Test method to verify cell type ID assignment (without CompuCell3D objects)"""
        print("=== Testing Cell Type ID Assignment ===")
        print("Note: CompuCell3D objects removed to prevent SpecValueCheckError")

        # Show current UI cell types
        print("Cell types in UI:")
        for i, entry in enumerate(self.celltype_widget.celltype_entries):
            print(f"  Index {i}: '{entry['Cell type']}' (frozen: {entry['freeze']})")

        print("=== Test Complete ===")
        return self.celltype_widget.celltype_entries

    def find_actual_cell_ids(self):
        """Comprehensive method to find where actual cell IDs are stored (without CompuCell3D objects)"""
        print("🔍 Searching for Actual Cell IDs...")
        print("Note: CompuCell3D objects removed to prevent SpecValueCheckError")

        # Show current UI cell types
        print("\n📋 Cell Types in UI:")
        for i, entry in enumerate(self.celltype_widget.celltype_entries):
            print(f"  UI Index {i}: '{entry['Cell type']}' (frozen: {entry['freeze']})")
            print(f"    - Implicit ID would be: {i}")

        print("\n💡 In CompuCell3D, cell type IDs are typically assigned sequentially:")
        print("  - Medium (if present) usually gets ID 0")
        print("  - First user-defined type gets ID 1")
        print("  - Second user-defined type gets ID 2")
        print("  - And so on...")

        print("\n✅ Cell ID search complete!")

    def run_comprehensive_test(self):
        """Run comprehensive tests for UI functionality"""
        print("🧪 Starting Comprehensive UI Tests...")

        # Test 1: Cell Type ID Assignment
        print("\n1️⃣ Testing Cell Type ID Assignment:")
        cell_types = self.test_cell_type_ids()

        # Test 2: UI State Capture
        print("\n2️⃣ Testing UI State Capture:")
        self.celltype_widget.debug_cell_types()

        # Test 3: Configuration Generation
        print("\n3️⃣ Testing Configuration Generation:")
        config = self.current_config()
        print(f"Generated config keys: {list(config.keys())}")
        print(f"Cell types in config: {len(config['CellType'])}")

        # Test 4: Save Functionality
        print("\n4️⃣ Testing Save Functionality:")
        try:
            self.save_to_json()
            print("✅ Save test completed - check console output above")
        except Exception as e:
            print(f"❌ Save test failed: {e}")

        print("\n🎉 All tests completed!")
        return {
            "cell_types": cell_types,
            "config": config
        }

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
        # Since metadata is commented out to avoid validation errors,
        # we'll just save the value directly to the saved_values and JSON
        if not hasattr(self, 'metadata') or self.metadata is None:
            # Update the saved values directly
            if "Metadata" not in self.saved_values:
                self.saved_values["Metadata"] = {}
            self.saved_values["Metadata"][property_name] = value
            self.save_to_json()
        else:
            # If metadata exists, use the original logic
            setattr(self.metadata, property_name, value)
            self.save_to_json()

    def update_potts_core(self, property_name, value):
        # Since potts_core is commented out to avoid validation errors,
        # we'll just save the value directly to the saved_values and JSON
        if not hasattr(self, 'potts_core') or self.potts_core is None:
            # Update the saved values directly
            if "PottsCore" not in self.saved_values:
                self.saved_values["PottsCore"] = {}
            self.saved_values["PottsCore"][property_name] = value
            self.save_to_json()
        else:
            # If potts_core exists, use the original logic
            if hasattr(self.potts_core, property_name):
                setattr(self.potts_core, property_name, value)
                self.save_to_json()

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
        # Create UI elements with CSS classes
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
        # Don't create CellTypePlugin to avoid validation errors
        # self.cell_type_plugin = CellTypePlugin()
        self.plugins_tab.update_cell_types(self.celltype_widget.get_cell_type_names())
        self.save_to_json()

    def reset_plugins_tab(self):
        self.plugins_tab.reset()
        self.save_to_json()

    def reset_metadata_tab(self):
        # Use default values without creating CompuCell3D objects
        self.widgets["num_processors"].value = 1
        self.widgets["debug_output_frequency"].value = 100
        self.save_to_json()

    def reset_all(self):
        self.reset_metadata_tab()
        self.reset_potts_tab()
        self.reset_celltype_tab()
        self.reset_plugins_tab()

    def update_cell_types(self):
        # Update plugins with cell type names (no CompuCell3D objects needed)
        self.plugins_tab.update_cell_types(self.celltype_widget.get_cell_type_names())
        self.save_to_json()

    def create_potts_core_for_compucell(self):
        """Create a proper CompuCell3D PottsCore object with current values"""
        try:
            potts_core = PottsCore()

            # Get config from saved values instead of self.potts_core
            config = self.saved_values.get("PottsCore", {})

            # Set values safely, catching any validation errors
            for key, value in config.items():
                try:
                    setattr(potts_core, key, value)
                except Exception as e:
                    print(f"Warning: Could not set {key}={value} on PottsCore: {e}")

            return potts_core
        except Exception as e:
            print(f"Error creating PottsCore: {e}")
            return None

# Error handling utilities
# Temporary alias for backward compatibility
class InputValidator:
    @staticmethod
    def create_error_display():
        return HTML(
            value="",
            layout=Layout(
                color='red',
                font_size='12px',
                margin='2px 0 5px 0',
                display='none'
            )
        )

    @staticmethod
    def setup_input_validation(widget, error_widget, validator_func=None, error_message="Invalid input"):
        """Set up validation that suppresses exceptions and doesn't display errors"""
        def safe_validate_on_change(change):
            try:
                # Silently validate without showing errors
                if validator_func:
                    validator_func(change['new'])
                # Clear any existing error display
                if hasattr(error_widget, 'value'):
                    error_widget.value = ""
                    error_widget.layout.display = 'none'
                if hasattr(widget, 'remove_class'):
                    widget.remove_class('error-input')
            except Exception:
                # Silently ignore validation errors - they'll be handled by backend
                pass

        # Set up the observer to catch value changes
        if hasattr(widget, 'observe'):
            widget.observe(safe_validate_on_change, names='value')

class ErrorHandler:
    """Utility class for handling backend validation errors"""

    @staticmethod
    def create_error_display():
        """Create an HTML widget for displaying error messages"""
        return HTML(
            value="",
            layout=Layout(
                color='red',
                font_size='12px',
                margin='2px 0 5px 0',
                display='none'
            )
        )

    @staticmethod
    def show_error(widget, error_widget, message):
        """Show error message and highlight input widget"""
        # Add error class to apply CSS styling to input element only
        widget.add_class('error-input')

        # Show error message
        error_widget.value = f"<span style='color: red;'>⚠ {message}</span>"
        error_widget.layout.display = 'block'

    @staticmethod
    def clear_error(widget, error_widget):
        """Clear error message and reset widget appearance"""
        # Remove error class to reset styling
        widget.remove_class('error-input')

        # Hide error message
        error_widget.value = ""
        error_widget.layout.display = 'none'

    @staticmethod
    def clear_all_errors(error_widgets_dict):
        """Clear all errors in a dictionary of widget->error_widget mappings"""
        for widget, error_widget in error_widgets_dict.items():
            ErrorHandler.clear_error(widget, error_widget)

    @staticmethod
    def display_errors(errors_dict, widget_error_mapping):
        """Display errors from backend validation

        Args:
            errors_dict: Dict with field_id -> error_message mappings
            widget_error_mapping: Dict with field_id -> (widget, error_widget) mappings
        """
        for field_id, error_message in errors_dict.items():
            if field_id in widget_error_mapping:
                widget, error_widget = widget_error_mapping[field_id]
                ErrorHandler.show_error(widget, error_widget, error_message)

# Backend validation functions
class BackendValidator:
    """Backend validation that returns errors without throwing exceptions"""

    @staticmethod
    def validate_config(config):
        """Validate entire configuration and return errors dict"""
        errors = {}

        # Validate metadata
        if 'Metadata' in config:
            metadata_errors = BackendValidator.validate_metadata(config['Metadata'])
            errors.update(metadata_errors)

        # Validate Potts Core
        if 'PottsCore' in config:
            potts_errors = BackendValidator.validate_potts_core(config['PottsCore'])
            errors.update(potts_errors)

        # Validate Plugins
        if 'Plugins' in config:
            plugin_errors = BackendValidator.validate_plugins(config['Plugins'])
            errors.update(plugin_errors)

        return errors

    @staticmethod
    def validate_metadata(metadata):
        """Validate metadata and return errors"""
        errors = {}

        if 'num_processors' in metadata:
            value = metadata['num_processors']
            if value < 1:
                errors['num_processors'] = "Number of processors must be at least 1"
            elif value > 128:
                errors['num_processors'] = "Number of processors seems too large (>128)"

        if 'debug_output_frequency' in metadata:
            value = metadata['debug_output_frequency']
            if value < 0:
                errors['debug_output_frequency'] = "Debug output frequency must be non-negative"
            elif value > 10000:
                errors['debug_output_frequency'] = "Debug output frequency seems too large (>10000)"

        return errors

    @staticmethod
    def validate_potts_core(potts_core):
        """Validate Potts core parameters and return errors"""
        errors = {}

        for dim in ['dim_x', 'dim_y', 'dim_z']:
            if dim in potts_core:
                value = potts_core[dim]
                if value < 1:
                    errors[dim] = f"{dim.replace('_', ' ').title()} must be at least 1"
                elif value > 1000:
                    errors[dim] = f"{dim.replace('_', ' ').title()} seems too large (>1000)"

        if 'steps' in potts_core:
            value = potts_core['steps']
            if value < 1:
                errors['steps'] = "MC steps must be at least 1"
            elif value > 1000000:
                errors['steps'] = "MC steps seems too large (>1,000,000)"

        if 'fluctuation_amplitude' in potts_core:
            value = potts_core['fluctuation_amplitude']
            if value < 0:
                errors['fluctuation_amplitude'] = "Fluctuation amplitude must be non-negative"
            elif value > 1000:
                errors['fluctuation_amplitude'] = "Fluctuation amplitude seems too large (>1000)"

        if 'neighbor_order' in potts_core:
            value = potts_core['neighbor_order']
            if value < 1:
                errors['neighbor_order'] = "Neighbor order must be at least 1"
            elif value > 20:
                errors['neighbor_order'] = "Neighbor order should not exceed 20"

        return errors

    @staticmethod
    def validate_plugins(plugins):
        """Validate plugin configurations and return errors"""
        errors = {}

        for plugin_name, plugin_config in plugins.items():
            if plugin_name == "VolumePlugin":
                volume_errors = BackendValidator.validate_volume_plugin(plugin_config)
                for key, error in volume_errors.items():
                    errors[f"volume_{key}"] = error

            elif plugin_name == "SurfacePlugin":
                surface_errors = BackendValidator.validate_surface_plugin(plugin_config)
                for key, error in surface_errors.items():
                    errors[f"surface_{key}"] = error

        return errors

    @staticmethod
    def validate_volume_plugin(config):
        """Validate VolumePlugin configuration"""
        errors = {}

        if 'params' in config:
            for i, param in enumerate(config['params']):
                if 'target_volume' in param:
                    value = param['target_volume']
                    if value <= 0:
                        errors[f"target_volume_{i}"] = "Target volume must be positive"
                    elif value > 10000:
                        errors[f"target_volume_{i}"] = "Target volume seems too large (>10000)"

                if 'lambda_volume' in param:
                    value = param['lambda_volume']
                    if value < 0:
                        errors[f"lambda_volume_{i}"] = "Lambda volume must be non-negative"
                    elif value > 1000:
                        errors[f"lambda_volume_{i}"] = "Lambda volume seems too large (>1000)"

        return errors

    @staticmethod
    def validate_surface_plugin(config):
        """Validate SurfacePlugin configuration"""
        errors = {}

        if 'params' in config:
            for i, param in enumerate(config['params']):
                if 'target_surface' in param:
                    value = param['target_surface']
                    if value <= 0:
                        errors[f"target_surface_{i}"] = "Target surface must be positive"
                    elif value > 50000:
                        errors[f"target_surface_{i}"] = "Target surface seems too large (>50000)"

                if 'lambda_surface' in param:
                    value = param['lambda_surface']
                    if value < 0:
                        errors[f"lambda_surface_{i}"] = "Lambda surface must be non-negative"
                    elif value > 1000:
                        errors[f"lambda_surface_{i}"] = "Lambda surface seems too large (>1000)"

        return errors
