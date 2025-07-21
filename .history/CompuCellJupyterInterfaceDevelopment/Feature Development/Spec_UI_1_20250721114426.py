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
/* Consistent tab container */
.tab-content-container {
    width: 100%;
    max-width: 1200px;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    padding: 15px;
    margin: 10px 0;
    background-color: #f9f9f9;
    overflow: auto;
    max-height: 600px;
}

/* Consistent section styling */
.config-section {
    margin-bottom: 20px;
    padding: 15px;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    background-color: white;
}

/* Tab header styling */
.widget-tab > .p-TabBar {
    background: #f5f5f5;
    border-bottom: 1px solid #e0e0e0;
}

.widget-tab > .p-TabBar > .p-TabBar-content {
    min-width: 100%;
    justify-content: flex-start;
}

.widget-tab > .p-TabBar > .p-TabBar-content > .p-TabBar-tab {
    padding: 8px 20px;
    border: 1px solid transparent;
    border-bottom: none;
    border-radius: 4px 4px 0 0;
    margin: 0 2px;
}

.widget-tab > .p-TabBar > .p-TabBar-content > .p-TabBar-tab.p-mod-current {
    background: white;
    border-color: #e0e0e0;
    border-bottom: 1px solid white;
    margin-bottom: -1px;
}

/* Consistent button styling */
.jupyter-button {
    margin: 5px;
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
        "CellType": CellTypePlugin().spec_dict, # still need to change to cell tpye spec_dict
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
    def __init__(self, plugin_name, plugin_class, saved_values, cell_types, parent_ui=None):
        self.plugin_name = plugin_name
        self.plugin_class = plugin_class
        self.default_instance = plugin_class()
        self.cell_types = cell_types
        self.widgets = {}
        self.output = Output()
        self.param_cache = {}
        self.parent_ui = parent_ui
        self.create_widgets(saved_values if saved_values else {})

    def create_widgets(self, saved_values):
        self.widgets["active"] = widgets.Checkbox(
            value=bool(saved_values),
            description=self.plugin_name,
            indent=False
        )

        self.widgets["config_container"] = VBox([], layout=Layout(
            padding='0',
            display='none' if not saved_values else 'block'
        ))

        def save_on_toggle(change):
            if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
                self.parent_ui.save_to_json()
        self.widgets["active"].observe(save_on_toggle, names='value')

        if self.plugin_name == "VolumePlugin":
            self.create_volume_widgets(saved_values)
        elif self.plugin_name == "SurfacePlugin":
            self.create_surface_widgets(saved_values)
        elif self.plugin_name == "AdhesionFlexPlugin":
            self.create_adhesion_widgets(saved_values)
        elif self.plugin_name == "ContactPlugin":
            self.create_contact_widgets(saved_values)
        elif self.plugin_name == "ChemotaxisPlugin":
            self.create_chemotaxis_widgets(saved_values)  # Fixed this line
        elif self.plugin_name == "BoundaryPixelTrackerPlugin":
            self.create_boundary_tracker_widgets(saved_values)
        elif self.plugin_name == "CurvaturePlugin":
            self.create_curvature_widgets(saved_values)
        elif self.plugin_name == "ExternalPotentialPlugin":
            self.create_external_potential_widgets(saved_values)
        elif self.plugin_name == "FocalPointPlasticityPlugin":
            self.create_focal_point_plasticity_widgets(saved_values)

        self.widgets["active"].observe(self.toggle_config_visibility, names='value')

    def toggle_config_visibility(self, change):
        if self.plugin_name in ["CurvaturePlugin", "ExternalPotentialPlugin", "FocalPointPlasticityPlugin"]:
            self.widgets["config_container"].layout.display = 'block'
        else:
            self.widgets["config_container"].layout.display = 'block' if change['new'] else 'none'

    # VolumePlugin widgets
    def create_volume_widgets(self, saved_values):
        default_params = VolumePlugin().spec_dict.get("params", [])
        default_map = {p["CellType"]: p for p in default_params} if default_params else {}
        saved_map = {p["CellType"]: p for p in saved_values.get("params", []) if "CellType" in p}
        rows = []
        for ct in self.cell_types:
            param = self.param_cache.get(ct) or saved_map.get(ct) or default_map.get(ct) or {
                "CellType": ct,
                "target_volume": 25.0,
                "lambda_volume": 2.0
            }
            row = {}
            row["cell_type"] = widgets.Label(value=ct, layout=widgets.Layout(width='120px'))
            row["target_volume"] = widgets.FloatText(
                value=param["target_volume"],
                min=0.0,
                description='',
                layout=widgets.Layout(width='120px')
            )
            row["lambda_volume"] = widgets.FloatText(
                value=param["lambda_volume"],
                min=0.0,
                description='',
                layout=widgets.Layout(width='120px')
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
        row_widgets = []
        for row in self.widgets["rows"]:
            row_box = HBox([
                row["cell_type"],
                widgets.Label("Target Volume:", layout=widgets.Layout(width='100px')),
                row["target_volume"],
                widgets.Label("Lambda Volume:", layout=widgets.Layout(width='100px')),
                row["lambda_volume"]
            ], layout=Layout(padding='4px 0 4px 12px'))
            row_widgets.append(row_box)
        self.widgets["config_container"].children = [VBox(row_widgets)]

    # SurfacePlugin widgets
    def create_surface_widgets(self, saved_values):
        default_params = SurfacePlugin().spec_dict.get("params", [])
        default_map = {p["CellType"]: p for p in default_params} if default_params else {}
        saved_map = {p["CellType"]: p for p in saved_values.get("params", []) if "CellType" in p}
        rows = []
        for ct in self.cell_types:
            param = self.param_cache.get(ct) or saved_map.get(ct) or default_map.get(ct) or {
                "CellType": ct,
                "target_surface": 100.0,
                "lambda_surface": 0.5
            }
            row = {}
            row["cell_type"] = widgets.Label(value=ct, layout=widgets.Layout(width='120px'))
            row["target_surface"] = widgets.FloatText(
                value=param["target_surface"],
                min=0.0,
                description='',
                layout=widgets.Layout(width='120px')
            )
            row["lambda_surface"] = widgets.FloatText(
                value=param["lambda_surface"],
                min=0.0,
                description='',
                layout=widgets.Layout(width='120px')
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

    # AdhesionFlexPlugin widgets
    def create_adhesion_widgets(self, saved_values):
        defaults = AdhesionFlexPlugin().spec_dict
        self.widgets["neighbor_order"] = widgets.IntText(
            value=saved_values.get("neighbor_order", defaults.get("neighbor_order", 2)),
            description='Neighbor Order:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["neighbor_order_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        self.widgets["max_distance"] = widgets.IntText(
            value=saved_values.get("max_distance", defaults.get("max_distance", 3)),
            min=1,
            description='Max Distance:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["max_distance_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        neighbor_box = VBox([self.widgets["neighbor_order"], self.widgets["neighbor_order_error"]])
        max_distance_box = VBox([self.widgets["max_distance"], self.widgets["max_distance_error"]])
        container = VBox([
            HBox([neighbor_box, max_distance_box])
        ])
        self.widgets["config_container"].children = [container]

    # ContactPlugin widgets
    def create_contact_widgets(self, saved_values):
        defaults = ContactPlugin().spec_dict
        self.widgets["neighbor_order"] = widgets.IntText(
            value=saved_values.get("neighbor_order", defaults.get("neighbor_order", 1)),
            min=1,
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
        self.widgets["config_container"].children = [container]

    # ChemotaxisPlugin widgets - ADDED MISSING METHOD
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
        defaults = BoundaryPixelTrackerPlugin().spec_dict
        self.widgets["neighbor_order"] = widgets.IntText(
            value=saved_values.get("neighbor_order", defaults.get("neighbor_order", 2)),
            description='Neighbor Order:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='250px')
        )
        self.widgets["neighbor_order_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        neighbor_box = VBox([
            self.widgets["neighbor_order"],
            self.widgets["neighbor_order_error"]
        ])
        self.widgets["config_container"].children = [neighbor_box]

    def create_curvature_widgets(self, saved_values):
        self.widgets["config_container"].children = []
        self.widgets["config_container"].layout.display = 'block'

    def create_external_potential_widgets(self, saved_values):
        self.widgets["config_container"].children = []
        self.widgets["config_container"].layout.display = 'block'

    def create_focal_point_plasticity_widgets(self, saved_values):
        self.widgets["config_container"].children = []
        self.widgets["config_container"].layout.display = 'block'

    def create_ui(self):
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

        for key, widget in self.widgets.items():
            if key not in ["active", "config_container", "rows", "add_btn"] and hasattr(widget, 'value'):
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
        self.widgets["active"].value = False
        self.widgets["config_container"].layout.display = 'none'
        default = self.default_instance.spec_dict

        if "rows" in self.widgets:
            self.widgets["rows"] = []
            if "params" in default:
                if self.plugin_name == "VolumePlugin":
                    self.create_volume_widgets(default)
                elif self.plugin_name == "SurfacePlugin":
                    self.create_surface_widgets(default)

        for key, widget in self.widgets.items():
            if key not in ["active", "config_container", "rows", "add_btn"] and key in default:
                widget.value = default[key]

    def _validate_plugin_input(self, plugin, field, value=None):
        widget_map = self.widgets
        error_widget_name = f"{field}_error"
        input_widget = widget_map.get(field)
        error_widget = widget_map.get(error_widget_name)
        if not input_widget or not error_widget:
            return
        error_widget.value = ""
        error_widget.layout.display = 'none'
        if hasattr(input_widget, 'remove_class'):
            input_widget.remove_class('error-input')
        try:
            plugin_class_map = {
                'AdhesionFlexPlugin': AdhesionFlexPlugin,
                'ChemotaxisPlugin': ChemotaxisPlugin,
                'ContactPlugin': ContactPlugin,
                'BoundaryPixelTrackerPlugin': BoundaryPixelTrackerPlugin
            }
            plugin_instance = plugin_class_map[plugin]()
            if value is not None:
                setattr(plugin_instance, field, value)
            else:
                setattr(plugin_instance, field, input_widget.value)
            if self.parent_ui and hasattr(self.parent_ui, 'potts_core') and hasattr(self.parent_ui, 'cell_type_plugin'):
                plugin_instance.validate(self.parent_ui.potts_core, self.parent_ui.cell_type_plugin)
            else:
                plugin_instance.validate()
            error_widget.value = ""
            error_widget.layout.display = 'none'
            if hasattr(input_widget, 'remove_class'):
                input_widget.remove_class('error-input')
        except Exception as e:
            msg = str(e)
            parts = [line.strip() for line in msg.split('\n') if line.strip()] or [msg]
            html = '<br>'.join(f'⚠️ {p}' for p in parts)
            error_widget.value = f'<span style="color: red; font-size: 12px;">{html}</span>'
            error_widget.layout.display = 'block'
            if hasattr(input_widget, 'add_class'):
                input_widget.add_class('error-input')

    def update_cell_types(self, cell_types):
        self.cell_types = cell_types
        if self.plugin_name == "VolumePlugin":
            self.create_volume_widgets({"params": [self.param_cache.get(ct, {"CellType": ct, "target_volume": 25.0, "lambda_volume": 2.0}) for ct in cell_types]})
        elif self.plugin_name == "SurfacePlugin":
            self.create_surface_widgets({"params": [self.param_cache.get(ct, {"CellType": ct, "target_surface": 100.0, "lambda_surface": 0.5}) for ct in cell_types]})

class PluginsTab:
    def __init__(self, saved_plugins, cell_types, parent_ui=None):
        self.widgets = {}
        self.plugin_widgets = {}
        self.cell_types = cell_types
        self.parent_ui = parent_ui
        self.create_widgets(saved_plugins or DEFAULTS["Plugins"])

    def create_widgets(self, saved_plugins):
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
            plugin_widget = PluginWidget(plugin_name, plugin_class, plugin_values, self.cell_types, parent_ui=self.parent_ui)
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
        ).add_class('button-spacing')

        # Connect reset button to save
        def on_reset_clicked(_):
            self.reset()
            if self.parent_ui and hasattr(self.parent_ui, 'save_to_json'):
                self.parent_ui.save_to_json()
                
        self.widgets["reset_button"].on_click(on_reset_clicked)

    def get_config(self):
        config = {}
        for plugin_name, widget in self.plugin_widgets.items():
            plugin_config = widget.get_config()
            if plugin_config is not None:
                config[plugin_name] = plugin_config
        return config

    def reset(self, _=None):
        for widget in self.plugin_widgets.values():
            widget.reset()

    def update_cell_types(self, cell_types):
        self.cell_types = cell_types
        for widget in self.plugin_widgets.values():
            widget.update_cell_types(cell_types)

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
        # Dimension inputs with max=101
        self.widgets["dim_x"] = widgets.BoundedIntText(
            value=saved_values.get("dim_x", self.defaults["dim_x"]),
            min=1, max=101, description='X Dimension:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["dim_y"] = widgets.BoundedIntText(
            value=saved_values.get("dim_y", self.defaults["dim_y"]),
            min=1, max=101, description='Y Dimension:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["dim_z"] = widgets.BoundedIntText(
            value=saved_values.get("dim_z", self.defaults["dim_z"]),
            min=1, max=101, description='Z Dimension:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )

        # Error display
        self.widgets["dim_x_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        self.widgets["dim_y_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        self.widgets["dim_z_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))

        # Core parameters
        self.widgets["steps"] = widgets.IntText(
            value=saved_values.get("steps", self.defaults["steps"]),
            min=0, description='MC Steps:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        self.widgets["steps_error"] = HTML(value="", layout=Layout(margin='2px 0 5px 0', display='none'))
        
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
        self.widgets["random_seed"] = widgets.IntText(
            value=saved_values.get("random_seed", self.defaults.get("random_seed", 0)),
            min=0, description='Random Seed:',
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
            "anneal": self.widgets["anneal"].value,
            "fluctuation_amplitude": self.widgets["fluctuation_amplitude"].value,
            "fluctuation_amplitude_function": self.widgets["fluctuation_amplitude_function"].value,
            "boundary_x": self.widgets["boundary_x"].value,
            "boundary_y": self.widgets["boundary_y"].value,
            "boundary_z": self.widgets["boundary_z"].value,
            "neighbor_order": self.widgets["neighbor_order"].value,
            "lattice_type": self.widgets["lattice_type"].value,
            "offset": self.widgets["offset"].value,
            "random_seed": self.widgets["random_seed"].value
        }

    def reset(self):
        for key, widget in self.widgets.items():
            if key != "reset_button" and key in self.defaults:
                widget.value = self.defaults[key]

    def create_ui(self):
    # Create left-aligned rows with consistent box styling
        dimensions_row = HBox([
            VBox([self.widgets["dim_x"], self.widgets["dim_x_error"]]),
            VBox([self.widgets["dim_y"], self.widgets["dim_y_error"]]),
            VBox([self.widgets["dim_z"], self.widgets["dim_z_error"]])
        ], layout=Layout(justify_content='flex-start', align_items='flex-start'))
        dimensions_row.add_class('vbox-row-spacing')

        core_params_row = HBox([
            VBox([self.widgets["steps"], self.widgets["steps_error"]]),
            VBox([self.widgets["anneal"], self.widgets["anneal_error"]]),
            VBox([self.widgets["fluctuation_amplitude"], self.widgets["fluctuation_amplitude_error"]])
        ], layout=Layout(justify_content='flex-start', align_items='flex-start'))
        core_params_row.add_class('vbox-row-spacing')
        
        fluct_function_row = HBox([
            self.widgets["fluctuation_amplitude_function"]
        ], layout=Layout(justify_content='flex-start', align_items='flex-start'))
        fluct_function_row.add_class('vbox-row-spacing')

        boundaries_row = HBox([
            self.widgets["boundary_x"],
            self.widgets["boundary_y"],
            self.widgets["boundary_z"]
        ], layout=Layout(justify_content='flex-start', align_items='flex-start'))
        boundaries_row.add_class('vbox-row-spacing')

        advanced_row1 = HBox([
            VBox([self.widgets["neighbor_order"], self.widgets["neighbor_order_error"]]),
            self.widgets["lattice_type"]
        ], layout=Layout(justify_content='flex-start', align_items='flex-start'))
        advanced_row1.add_class('vbox-row-spacing')

        advanced_row2 = HBox([
            self.widgets["offset"],
            self.widgets["random_seed"]
        ], layout=Layout(justify_content='flex-start', align_items='flex-start'))
        advanced_row2.add_class('vbox-row-spacing')

        # Wrap in consistent config box
        return VBox([
            HTML("<b>Potts Core Parameters</b>", layout=Layout(margin='0 0 10px 0')),
            dimensions_row,
            core_params_row,
            fluct_function_row,
            HTML("<b>Boundary Conditions:</b>", layout=Layout(margin='10px 0 5px 0')),
            boundaries_row,
            HTML("<b>Advanced Settings:</b>", layout=Layout(margin='10px 0 5px 0')),
            advanced_row1,
            advanced_row2,
            HBox([self.widgets["reset_button"]], layout=Layout(margin='10px 0 0 0'))
        ], layout=Layout(
            align_items='flex-start',
            padding='10px'
        )).add_class('config-section')

class CellTypeWidget:
    def __init__(self, saved_entries, on_change=None):
        self.on_change = on_change
        self.next_id = 1  # Start at 1 since Medium is 0
        self.celltype_entries = []
        
        # Load saved entries or use defaults
        entries = saved_entries or DEFAULTS["CellType"]
        for entry in entries:
            if isinstance(entry, dict):
                self.add_entry(entry["Cell type"], entry.get("id", self.next_id), entry.get("freeze", False))
            else:
                self.add_entry(entry, self.next_id, False)
        
        # Ensure Medium always exists with ID 0
        if not any(entry["Cell type"] == "Medium" for entry in self.celltype_entries):
            self.add_entry("Medium", 0, False)

        self.widgets = {}
        self.create_widgets()
        self.setup_event_handlers()
        self.update_celltype_display()

    def add_entry(self, name, type_id, freeze):
        # Enforce ID uniqueness
        if any(entry["id"] == type_id for entry in self.celltype_entries):
            available_id = max(entry["id"] for entry in self.celltype_entries) + 1
            type_id = available_id
            self.next_id = available_id + 1
        
        # Enforce Medium always has ID 0
        if name == "Medium":
            type_id = 0
        
        self.celltype_entries.append({
            "Cell type": name,
            "id": type_id,
            "freeze": freeze
        })
        
        # Update next available ID
        if type_id >= self.next_id:
            self.next_id = type_id + 1

    def create_widgets(self):
        self.widgets["display_box"] = VBox(layout=Layout(padding='10px'))
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
        if any(entry["Cell type"] == name for entry in self.celltype_entries):
            self.widgets["name"].value = ""
            self.widgets["name"].placeholder = f"{name} already exists!"
            return

        self.add_entry(name, self.next_id, self.widgets["freeze"].value)
        self.update_celltype_display()
        self.widgets["name"].value = ""
        self.widgets["name"].placeholder = "Cell type name"
        if self.on_change:
            self.on_change()

    def update_celltype_display(self):
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
            
            # Freeze column - now using Checkbox instead of Label
            freeze_chk = Checkbox(
                value=entry.get('freeze', False),
                indent=False,
                layout=Layout(border=border_style, padding='2px 8px', width='auto')
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
                    description="Cannot remove",
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
        return self.celltype_entries.copy()

    def get_cell_type_names(self):
        return [entry["Cell type"] for entry in self.celltype_entries]

    def reset(self, _=None):
        self.celltype_entries = []
        self.next_id = 1  # Reset to 1 (Medium will be 0)
        for entry in DEFAULTS["CellType"]:
            self.add_entry(entry["Cell type"], entry["id"], entry.get("freeze", False))
        self.update_celltype_display()

    def create_ui(self):
        input_row = HBox([
            self.widgets["name"],
            self.widgets["freeze"],
            self.widgets["add_button"]
        ], layout=Layout(justify_content='flex-start', margin='10px 0'))
        
        reset_button_box = HBox([self.widgets["reset_button"]], 
                               layout=Layout(justify_content='flex-start', margin='10px 0'))
        
        return VBox([
            HTML("<b>Cell Types</b>", layout=Layout(margin='0 0 10px 0')),
            self.widgets["display_box"],
            input_row,
            reset_button_box
        ], layout=Layout(
            align_items='flex-start',
            padding='10px'
        )).add_class('config-section')

class SpecificationSetupUI:
    def __init__(self):
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
            parent_ui=self
        )
        self.create_ui()
        self.setup_event_handlers()
        self.check_ready_state()

    def cell_types_changed(self):
        self.update_plugin_cell_types()
        self.save_to_json()
        self.check_ready_state()

    def update_plugin_cell_types(self):
        cell_types = self.celltype_widget.get_config()
        cell_type_names = [entry["Cell type"] for entry in cell_types]
        volume_widget = self.plugins_tab.plugin_widgets.get("VolumePlugin")
        if volume_widget:
            volume_widget.update_cell_types(cell_type_names)
        surface_widget = self.plugins_tab.plugin_widgets.get("SurfacePlugin")
        if surface_widget:
            surface_widget.update_cell_types(cell_type_names)

    def apply_saved_values(self):
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
        self.widgets["run_button"] = Button(
            description="Ready to run simulation",
            button_style='success',
            style={'description_width': 'initial'},
            disabled=True
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
        if config:
            with open(SAVE_FILE, 'w') as f:
                json.dump(config, f, indent=4)
        self.check_ready_state()

    def setup_event_handlers(self):
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
        self.widgets["run_button"].on_click(
            lambda _: self.run_simulation()
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

    def check_ready_state(self):
        required_ready = (
            self.potts_widget.widgets["dim_x"].value > 0 and
            self.potts_widget.widgets["dim_y"].value > 0 and
            self.potts_widget.widgets["dim_z"].value > 0 and
            self.potts_widget.widgets["steps"].value > 0 and
            len(self.celltype_widget.celltype_entries) > 0
        )
        self.widgets["run_button"].disabled = not required_ready

    def update_metadata(self, property_name, value):
        try:
            setattr(self.metadata, property_name, value)
            self.save_to_json()
            self.clear_metadata_error(property_name)
        except Exception as e:
            self.show_metadata_error(property_name, str(e))

    def show_metadata_error(self, property_name, message):
        error_widget_name = f"{property_name}_error"
        input_widget = self.widgets.get(property_name)
        if error_widget_name in self.widgets:
            error_widget = self.widgets[error_widget_name]
            error_widget.value = f'<span style="color: red; font-size: 12px;">⚠️ {message}</span>'
            error_widget.layout.display = 'block'
        if input_widget is not None and hasattr(input_widget, 'add_class'):
            input_widget.add_class('error-input')

    def clear_metadata_error(self, property_name):
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
            setattr(self.potts_core, property_name, value)
            self.save_to_json()
            self.clear_potts_error(property_name)
        except Exception as e:
            self.show_potts_error(property_name, str(e))

    def show_potts_error(self, property_name, message):
        error_widget_name = f"{property_name}_error"
        input_widget = self.potts_widget.widgets.get(property_name)
        if error_widget_name in self.potts_widget.widgets:
            error_widget = self.potts_widget.widgets[error_widget_name]
            error_widget.value = f'<span style="color: red; font-size: 12px;">⚠️ {message}</span>'
            error_widget.layout.display = 'block'
        if input_widget is not None and hasattr(input_widget, 'add_class'):
            input_widget.add_class('error-input')

    def clear_potts_error(self, property_name):
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

        run_button_box = HBox([self.widgets["run_button"]], layout=Layout(justify_content='flex-end'))
        
        display(VBox([
            tabs,
            run_button_box
        ], layout=Layout(align_items='flex-start')))

    def create_metadata_tab(self):
        num_processors_box = VBox([
            self.widgets["num_processors"],
            self.widgets["num_processors_error"]
        ], layout=Layout(align_items='flex-start'))
        
        debug_frequency_box = VBox([
            self.widgets["debug_output_frequency"],
            self.widgets["debug_output_frequency_error"]
        ], layout=Layout(align_items='flex-start'))
        
        reset_button_box = HBox([self.widgets["reset_metadata_button"]], layout=Layout(justify_content='flex-start'))
        
        return VBox([
            HTML("<b>Simulation Metadata</b>"),
            num_processors_box,
            debug_frequency_box,
            reset_button_box
        ], layout=Layout(align_items='flex-start'))

    def reset_potts_tab(self):
        self.potts_widget.reset()
        for prop, value in PottsCore().spec_dict.items():
            if hasattr(self.potts_core, prop):
                setattr(self.potts_core, prop, value)
        self.save_to_json()
        self.check_ready_state()

    def reset_celltype_tab(self):
        self.celltype_widget.reset()
        self.cell_type_plugin = CellTypePlugin()
        self.plugins_tab.update_cell_types(self.celltype_widget.get_cell_type_names())
        self.save_to_json()
        self.check_ready_state()

    def reset_plugins_tab(self):
        self.plugins_tab = PluginsTab(
            DEFAULTS["Plugins"],
            self.celltype_widget.get_cell_type_names(),
            parent_ui=self
        )
        self.save_to_json()
        self.check_ready_state()

    def reset_metadata_tab(self):
        self.widgets["num_processors"].value = Metadata().num_processors
        self.widgets["debug_output_frequency"].value = Metadata().debug_output_frequency
        self.metadata = Metadata()
        self.save_to_json()
        self.check_ready_state()

    def update_cell_types(self):
        self.cell_type_plugin = CellTypePlugin()
        for entry in self.celltype_widget.celltype_entries:
            self.cell_type_plugin.cell_type_append(
                entry["Cell type"],
                frozen=entry["freeze"]
            )
        self.plugins_tab.update_cell_types(self.celltype_widget.get_cell_type_names())
        self.save_to_json()
        self.check_ready_state()

    def run_simulation(self):
        print("Simulation is ready to run with current configuration")

# Create and display the UI
ui = SpecificationSetupUI()