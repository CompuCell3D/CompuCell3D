import math
import json
from IPython.core.display import display

from ipywidgets.widgets.widget_box import HBox, VBox
from ipywidgets import Layout

from cc3d.core.GraphicsUtils.JupyterWidgetInterface import JupyterWidgetInterface
from cc3d.core.GraphicsOffScreen.primitives import Color
from cc3d.CompuCellSetup.simulation_utils import extract_type_names_and_ids
from itertools import product
import cc3d.CompuCellSetup

# monkey patch the json encoder to use to_json function if available when serializing
from json import JSONEncoder
def _default(self, obj):
    return getattr(obj.__class__, "to_json", _default.default)(obj)
_default.default = JSONEncoder().default
JSONEncoder.default = _default


def split_half(arr):
    """
    Utility function to split array in half
    Returns 2 arrays
    """
    n = len(arr)
    half = int(math.ceil(n/2))
    return arr[0:half], arr[half:n+1]


class JupyterControlPanel:
    """
    Create the ipywidgets menus to edit view settings for JupyterGraphicsFrameClients
    """

    def __init__(self, rows=1, cols=1):
        self.wi = JupyterWidgetInterface()
        # _clients_data is data associated with each frame client in grid
        # list of lists (grid) of dicts {'client': frameclient, 'wi data': wi values, 'active': bool}
        self._clients_data = [[{} for j in range(cols)] for i in range(rows)]

    @property
    def rows(self) -> int:
        """Number of client rows attached to this control panel"""
        return len(self._clients_data)

    @property
    def cols(self) -> int:
        """Number of client columns attached to this control panel"""
        if self.rows == 0:
            return 0
        return len(self._clients_data[0])

    @property
    def _active_clients(self):
        active_clients = []
        for row in self._clients_data:
            for data in row:
                if 'active' in data and data['active']:
                    active_clients.append(data['client'])
        return active_clients

    def set_frame(self, frameclient, row: int = 0, col: int = 0):
        self._clients_data[row][col] = dict(client=frameclient, wi_data={}, active=True)

    def show(self):
        """
        Display the widgets and each frame client
        """
        if len(self._active_clients) > 0:
            self._create_control_panel()
        else:
            print('No frame clients have been added. Use .set_frame(frameclient, row, col)')

    def set_disabled(self, value=True):
        self.wi.set_all_disabled(value)

    def _create_control_panel(self):
        """Create view controls (ipywidgets)"""
        firstframe = self._active_clients[0].frame
        self._make_visualization_tab(firstframe)
        self._make_camera_tab(firstframe)
        self._make_colors_tab(firstframe)
        self._make_field_specific_options_tab(firstframe)

        # on creation, all clients have the same settings
        for i, row in enumerate(self._clients_data):
            for j in range(len(row)):
                self._clients_data[i][j]['wi_data'] = self.wi.values.copy()

        left = self._get_frames_widget()
        right = self.wi.tabs
        left.layout = Layout(width='20%')
        right.layout = Layout(width='80%')
        hbox = HBox([left, right])
        display(hbox)

    def _callback(self, fun):
        """
        Callback wrapper for a commonly used callback format
        """
        def f(keyname, value):
            for i, row in enumerate(self._clients_data):
                for j, data in enumerate(row):
                    if data['active']:
                        client = data['client']
                        fun(client, keyname, value)
                        self._update_client_data(i, j)
                        client.draw()
            self._check_conflicting_data()
        return f

    def _get_frames_widget(self):

        def set_active_client(keyname, value):
            # grid widget returns value of list of lists with bool
            first_wi_data = None
            for i, row in enumerate(value):
                for j, is_active in enumerate(row):
                    if is_active:
                        if not first_wi_data:
                            first_wi_data = self._clients_data[i][j]['wi_data']
                    self._clients_data[i][j]['active'] = is_active
                    self._clients_data[i][j]['client'].frame.border_selected = is_active
                    self._clients_data[i][j]['client'].draw()
            if first_wi_data:
                self.wi.update_values(first_wi_data)
            self._check_conflicting_data()

        return self.wi.add_grid('selected frames', rows=self.rows, cols=self.cols, value=True, callback=set_active_client)

    def _make_camera_tab(self, defaultframe):

        def set_drawing_style(c, keyname, value):
            c.set_drawing_style(value)
        self.wi.add_select('drawing style', options=['2D', '3D'], callback=self._callback(set_drawing_style))

        def set_camera_sync(keyname, value):
            active_clients = self._active_clients
            if len(active_clients) < 2:
                return

            ac0 = active_clients[0]
            if value:
                for ac in active_clients[1:]:
                    ac0.sync_cameras(ac)
            else:
                for ac in active_clients:
                    ac.unsync_camera()

        self.wi.add_toggle('camera sync', callback=set_camera_sync, value=False)

        def set_depth_x(c, keyname, value):
            c.set_plane('x', value)

        def set_depth_y(c, keyname, value):
            c.set_plane('y', value)

        def set_depth_z(c, keyname, value):
            c.set_plane('z', value)

        self.wi.add_int('depth x', 0, -100, 100, 1, self._callback(set_depth_x))
        self.wi.add_int('depth y', 0, -100, 100, 1, self._callback(set_depth_y))
        self.wi.add_int('depth z', 0, -100, 100, 1, self._callback(set_depth_z))

        xyz = ['x', 'y', 'z']
        fields = ['depth']
        names = [i+' '+j for i, j in product(fields, xyz)]
        self.wi.make_tab('Camera', ['drawing style', 'camera sync'], names)

    def _make_visualization_tab(self, defaultframe):
        # callback functions
        def toggle_bounding_box(c, keyname, value):
            c.frame.bounding_box_on = value

        def toggle_cell_borders(c, keyname, value):
            c.frame.cell_borders_on = value

        def toggle_cell_glyphs(c, keyname, value):
            c.frame.cell_glyphs_on = value

        def toggle_cells(c, keyname, value):
            c.frame.cells_on = value

        def toggle_cluster_borders(c, keyname, value):
            c.frame.cluster_borders_on = value

        def toggle_fpp_links(c, keyname, value):
            c.frame.fpp_links_on = value

        def toggle_lattice_axes_labels(c, keyname, value):
            c.frame.lattice_axes_labels_on = value

        def toggle_lattice_axes(c, keyname, value):
            c.frame.lattice_axes_on = value

        frame_options = [
            ('bounding box', toggle_bounding_box, defaultframe.bounding_box_on),
            ('cell borders', toggle_cell_borders, defaultframe.cell_borders_on),
            ('cell glyphs', toggle_cell_glyphs, defaultframe.cell_glyphs_on),
            ('cells', toggle_cells, defaultframe.cells_on),
            ('cluster borders', toggle_cluster_borders, defaultframe.cluster_borders_on),
            ('fpp links', toggle_fpp_links, defaultframe.fpp_links_on),
            ('lattice axes labels', toggle_lattice_axes_labels, defaultframe.lattice_axes_labels_on),
            ('lattice axes', toggle_lattice_axes, defaultframe.lattice_axes_on)
        ]
        for (field, func, value) in frame_options:
            self.wi.add_toggle(field, callback=self._callback(func), value=value)
        widget_names = [field for (field, _, _) in frame_options]

        # --- FIELD ---
        def toggle_field(c, keyname, value):
            c.frame.field_name = value

        options = defaultframe.fieldTypes.keys()
        self.wi.add_select('field', options=options, callback=self._callback(toggle_field))
        widget_names += ['field']

        # --- Make the widget tab
        options1, options2 = split_half(widget_names)
        self.wi.make_tab('Visualization', options1, options2)

    def _make_colors_tab(self, defaultframe):
        # callback
        def set_color(c, keyname, value):
            n = int(keyname[0])  # color number
            c.frame.colormap[n] = Color.from_str_rgb(value)
            c.frame.config.setSetting('TypeColorMap', c.frame.colormap)

        cell_type_colors = extract_type_names_and_ids()
        names = [f'{index} {name}' for index,name in cell_type_colors.items()]

        default_colors = list(defaultframe.colormap.values())
        for (index, name) in cell_type_colors.items():
            self.wi.add_color(f'{index} {name}', value=str(default_colors[index]), callback=self._callback(set_color))

        colors1, colors2 = split_half(names)

        self.wi.make_tab('Cell Colors', colors1, colors2)

    def _make_field_specific_options_tab(self, defaultframe):

        concentration_field_names = cc3d.CompuCellSetup.persistent_globals.simulator.getConcentrationFieldNameVector()

        def _make_callback(field, settingname):
            def cb(c, keyname, value):
                data = c.frame.config.getSetting(settingname)
                data[field] = value
                c.frame.config.setSetting(settingname, data) 
            return cb

        def _set_listener(keyname, target, field, callback):
            def listener(kn, value):
                callback(target+field, value)
            self.wi.add_listener(keyname+field, listener)
            self.wi._call_listeners(keyname+field)

        def _disabled_callback(target, value):
            self.wi.set_disabled(target, not value)

        def _min_lessthan_max_callback(target, value):
            if value < self.wi.data[target]['widget'].value:
                self.wi.data[target]['widget'].value = value

        def _max_greaterthan_min_callback(target, value):
            if value > self.wi.data[target]['widget'].value:
                self.wi.data[target]['widget'].value = value
        
        def _get_default(field, settingname):
            return defaultframe.config.getSetting(settingname)[field]

        list_of_field_widgets = []
        for field in concentration_field_names:
            field_widgets = []

            settingname = 'ContoursOn'
            w_on = self.wi.add_toggle('contours on',
                                      keyname=settingname+field,
                                      value=_get_default(field, settingname),
                                      callback=self._callback(_make_callback(field, settingname)))
            settingname = 'NumberOfContourLines'
            w = self.wi.add_int('contour lines',
                                keyname=settingname+field,
                                value=_get_default(field, settingname),
                                callback=self._callback(_make_callback(field, settingname)),
                                min=0,
                                max=100)
            _set_listener('ContoursOn', 'NumberOfContourLines', field, _disabled_callback)
            field_widgets.append(HBox([w_on, w]))

            settingname = 'LegendEnable'
            w = self.wi.add_toggle('enable legend',
                                   keyname=settingname+field,
                                   value=_get_default(field, settingname),
                                   callback=self._callback(_make_callback(field, settingname)))
            field_widgets.append(w)

            settingname = 'DisplayMinMaxInfo'
            w = self.wi.add_toggle('show min/max info',
                                   keyname=settingname+field,
                                   value=_get_default(field, settingname),
                                   callback=self._callback(_make_callback(field, settingname)))
            field_widgets.append(w)

            settingname = 'MinRangeFixed'
            w_on = self.wi.add_toggle('use fixed min range',
                                      keyname=settingname+field,
                                      value=_get_default(field, settingname),
                                      callback=self._callback(_make_callback(field, settingname)))
            settingname = 'MinRange'
            w = self.wi.add_float('min value',
                                  keyname=settingname+field,
                                  value=_get_default(field, settingname),
                                  callback=self._callback(_make_callback(field, settingname)))
            _set_listener('MinRangeFixed', 'MinRange', field, _disabled_callback)
            field_widgets.append(HBox([w_on, w]))

            settingname = 'MaxRangeFixed'
            w_on = self.wi.add_toggle('use fixed max range',
                                      keyname=settingname+field,
                                      value=_get_default(field, settingname),
                                      callback=self._callback(_make_callback(field, settingname)))
            settingname = 'MaxRange'
            w = self.wi.add_float('max value',
                                  keyname=settingname+field,
                                  value=_get_default(field, settingname),
                                  callback=self._callback(_make_callback(field, settingname)))
            _set_listener('MaxRangeFixed', 'MaxRange', field, _disabled_callback)
            field_widgets.append(HBox([w_on, w]))

            _set_listener('MaxRange', 'MinRange', field, _min_lessthan_max_callback)
            _set_listener('MinRange', 'MaxRange', field, _max_greaterthan_min_callback)

            list_of_field_widgets.append(VBox(field_widgets))

        self.wi.add_accordion('accordion', list_of_field_widgets, concentration_field_names)

        # --- Make the widget tab
        self.wi.make_tab('Field Specific Options', ['accordion'])

    def _update_client_data(self, row, col):
        """
        Set internal data associated with each client
        """
        self._clients_data[row][col]['wi_data'] = self.wi.values.copy()

    def _check_conflicting_data(self):
        """
        Checks for any conflicting values for active clients for each field
        """
        active_client_datas = []
        for row in self._clients_data:
            for data in row:
                if data['active']:
                    active_client_datas.append(data['wi_data'])
        
        if len(active_client_datas) > 0:
            self.wi.set_all_disabled(False)
            for key, val in active_client_datas[0].items():
                if key in ['selected frames', 'field']:
                    continue
                conflict = False
                for cd in active_client_datas:
                    if val != cd[key]:
                        conflict = True

                w = self.wi.data[key]['widget']
                if w and hasattr(w, 'description'):
                    marked = '*' in w.description

                    if conflict and not marked:
                        w.description = key+'*'
                    elif not conflict and marked:
                        w.description = w.description.replace('*', '')
        else:
            self.wi.set_all_disabled(True)


class JupyterSettingsPanel:
    """
    Create the ipywidgets menus to edit view settings for JupyterGraphicsFrameClients
    """

    def __init__(self, config, import_callback):
        self.wi = JupyterWidgetInterface()
        self.config = config
        self.import_callback = import_callback

    def show(self):
        """
        Display the widgets and each frame client
        """
        self._create_panel()

    def _create_panel(self):
        """Create view controls (ipywidgets)"""
        self._make_visualization_tab()
        self._make_colors_tab()
        self._make_json_tab()

        display(self.wi.tabs)

    def _make_visualization_tab(self):
        # callback
        frame_options = [
            ('bounding box',
             self.config.getSetting('BoundingBoxOn'),
             lambda v: self.config.setSetting('BoundingBoxOn', v)),
            ('cell borders',
             self.config.getSetting('CellBordersOn'),
             lambda v: self.config.setSetting('CellBordersOn', v)),
            ('cell glyphs',
             self.config.getSetting('CellGlyphsOn'),
             lambda v: self.config.setSetting('CellGlyphsOn', v)),
            ('cells',
             self.config.getSetting('CellsOn'),
             lambda v: self.config.setSetting('CellsOn', v)),
            ('cluster borders',
             self.config.getSetting('ClusterBordersOn'),
             lambda v: self.config.setSetting('ClusterBordersOn', v)),
            ('fpp links',
             self.config.getSetting('FPPLinksOn'),
             lambda v: self.config.setSetting('FPPLinksOn', v)),
            ('lattice axes labels',
             self.config.getSetting('ShowAxes'),
             lambda v: self.config.setSetting('ShowAxes', v)),
            ('lattice axes',
             self.config.getSetting('ShowHorizontalAxesLabels') or self.config.getSetting('ShowVerticalAxesLabels'),
             lambda v: self.config.setSetting('ShowHorizontalAxesLabels', v))
        ]
        for (field, value, cb) in frame_options:
            self.wi.add_toggle(field, value=value, callback=cb)
        frame_option_widget_names = [field for (field,_,_) in frame_options]
        options1, options2 = split_half(frame_option_widget_names)
        self.wi.make_tab('Visualization', options1, options2)

    def _make_colors_tab(self):
        colormap = self.config.getSetting('TypeColorMap')

        # callback
        def set_color(field_name, value):
            n = int(field_name[0])  # color number
            colormap[n] = Color.from_str_rgb(value)
            self.config.setSetting('TypeColorMap', colormap)

        colorpicker_names_values = [(f'cell color {k}', str(v)) for (k,v) in colormap.items()]
        for (name, value) in colorpicker_names_values:
            self.wi.add_color(name, value=value, callback=set_color)
        colorpicker_names = [name for (name,value) in colorpicker_names_values]
        colors1, colors2 = split_half(colorpicker_names)
        self.wi.make_tab('Cell Colors', colors1, colors2)

    def _make_json_tab(self):

        def export_callback():
            datastring = json.dumps(self.config.config_data)
            return datastring

        def import_callback_wrapper(data):
            data = list(data.values())[0]['content']
            data = json.loads(data.decode("utf-8"))
            self.import_callback(data)

        self.wi.add_download_button('export as json', filename='config.json', filecontents=export_callback)
        self.wi.add_upload_button('import json', accept='.json', callback=import_callback_wrapper)

        self.wi.make_tab('Import/Export', ['export as json', 'import json'])
