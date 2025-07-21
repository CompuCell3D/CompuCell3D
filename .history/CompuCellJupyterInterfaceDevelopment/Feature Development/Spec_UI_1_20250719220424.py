

class CellTypeWidget:
    def __init__(self, saved_entries, on_change=None):
        self.on_change = on_change
        self.next_id = 0
        self.celltype_entries =[]

        entries = saved_entries or DEFAULTS["CellType"]
        for entry in entries:
            if isinstance(entry, dict):
                self.add_entry(entry["Cell type"], entry.get("id", self.next_id), entry.get("freeze", False))
            else:
                self.add_entry(entry, self.next_id, False)

        if not any(entry["Cell type"] == "Medium" for entry in self.celltype_entries):
            self.add_entry("Medium", 0, False)

        self.widgets = {}
        self.create_widgets()
        self.setup_event_handlers()
        self.update_celltype_display()

    def add_entry(self, name, type_id, freeze):
        self.celltype_entries.append({
            "Cell type": name,
            "id": type_id,
            "freeze": freeze
        })
        self.next_id = max(self.next_id, type_id) + 1

    def create_widgets(self):
        self.widgets["display_box"] = VBox(
            layout=Layout(padding='10px')
        )
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
            HTML(f"<b style='display:block; padding:2px 8px;'>Frozen</b>", layout=Layout(border=f'0 0 {header_border} 0')),
            HTML(f"<b style='display:block; padding:2px 8px;'>Remove</b>", layout=Layout(border=f'0 0 {header_border} 0'))
        ]
        grid = GridspecLayout(n + 1, 4, grid_gap="0px")
        for j, h in enumerate(header):
            grid[0, j] = h

        for i, entry in enumerate(self.celltype_entries):
            border_style = f'0 0 {row_border} 0' if i < n - 1 else '0'
            grid[i + 1, 0] = Label(str(entry['id']), layout=Layout(border=border_style, padding='2px 8px'))
            grid[i + 1, 1] = Label(str(entry['Cell type']), layout=Layout(border=border_style, padding='2px 8px'))
            grid[i + 1, 2] = Label("Yes" if entry.get('freeze', False) else "No", layout=Layout(border=border_style, padding='2px 8px'))
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
            grid[i + 1, 3] = remove_btn

        self.widgets["display_box"].children = [grid]

    def get_config(self):
        return self.celltype_entries.copy()

    def get_cell_type_names(self):
        return [entry["Cell type"] for entry in self.celltype_entries]

    def reset(self, _=None):
        self.celltype_entries = []
        self.next_id = 0
        for entry in DEFAULTS["CellType"]:
            self.add_entry(entry["Cell type"], entry["id"], entry.get("freeze", False))
        self.update_celltype_display()

    def create_ui(self):
        input_row = HBox([
            self.widgets["name"],
            self.widgets["freeze"],
            self.widgets["add_button"]
        ], layout=Layout(justify_content='center'))
        reset_button_box = HBox([self.widgets["reset_button"]], layout=Layout(justify_content='center'))
        return VBox([
            HTML("<b>Cell Types</b>", layout=Layout(display='flex', justify_content='center')),
            self.widgets["display_box"],
            input_row,
            reset_button_box
        ])


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
        self.widgets["reset_button"] = Button(
            description="Reset All to Defaults",
            button_style='danger'
        )
        self.widgets["run_button"] = Button(
            description="Ready to run simulation",
            button_style='success',
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
        self.widgets["reset_button"].on_click(
            lambda _: self.reset_all()
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

        reset_button_box = HBox([self.widgets["reset_button"]], layout=Layout(justify_content='center'))
        run_button_box = HBox([self.widgets["run_button"]], layout=Layout(justify_content='center'))
        button_row = HBox([reset_button_box, run_button_box], layout=Layout(justify_content='center'))
        button_row.add_class('vbox-row-spacing')

        display(VBox([
            tabs,
            button_row
        ]))

    def create_metadata_tab(self):
        num_processors_box = VBox([
            self.widgets["num_processors"],
            self.widgets["num_processors_error"]
        ], layout=Layout(align_items='center'))
        
        debug_frequency_box = VBox([
            self.widgets["debug_output_frequency"],
            self.widgets["debug_output_frequency_error"]
        ], layout=Layout(align_items='center'))
        
        reset_button_box = HBox([self.widgets["reset_metadata_button"]], layout=Layout(justify_content='center'))
        
        return VBox([
            HTML("<b>Simulation Metadata</b>", layout=Layout(display='flex', justify_content='center')),
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

    def reset_all(self):
        self.reset_metadata_tab()
        self.reset_potts_tab()
        self.reset_celltype_tab()
        self.reset_plugins_tab()
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