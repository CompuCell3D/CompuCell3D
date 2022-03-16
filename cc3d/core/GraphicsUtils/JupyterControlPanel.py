from calendar import c
import math

# from cc3d.core.GraphicsUtils.JupyterGraphicsFrameWidget import CC3DJupyterGraphicsFrameGrid
from cc3d.core.GraphicsUtils.JupyterWidgetInterface import JupyterWidgetInterface
from cc3d.core.GraphicsOffScreen.primitives import Color
from itertools import product


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
        # self.grid = CC3DJupyterGraphicsFrameGrid(rows=rows, cols=cols)
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
                if data['active']:
                    active_clients.append(data['client'])
        return active_clients

    def set_frame(self, frameclient, row: int = 0, col: int = 0):
        # self.grid.set_frame(frameclient, row, col)
        self._clients_data[row][col] = dict(client=frameclient, wi_data={}, active=True)

    def show(self):
        """
        Display the widgets and each frame client
        """
        if len(self._active_clients) > 0:
            # self.grid.sync_cameras()
            self._create_control_panel()
            # self.grid.show()
        else:
            print('No frame clients have been added. Use .set_frame(frameclient, row, col)')

    def set_disabled(self, value=True):
        self.wi.set_disabled(value)

    def _create_control_panel(self):
        """Create view controls (ipywidgets)"""
        firstframe = self._active_clients[0].frame
        self._make_client_select_tab(firstframe)
        self._make_visualization_tab(firstframe)
        self._make_camera_tab(firstframe)
        self._make_colors_tab(firstframe)

        # on creation, all clients have the same settings
        for i, row in enumerate(self._clients_data):
            for j in range(len(row)):
                self._clients_data[i][j]['wi_data'] = self.wi.values.copy()

    def _callback(self, fun):
        """
        Callback wrapper for a commonly used callback format
        """
        def f(value):
            for i, row in enumerate(self._clients_data):
                for j, data in enumerate(row):
                    if data['active']:
                        client = data['client']
                        fun(client, value)
                        self._update_client_data(i, j)
                        client.draw()
            self._check_conflicting_data()
        return f

    def _make_client_select_tab(self, defaultframe):

        def set_active_client(value):
            # grid widget returns value of list of lists with bool
            first_wi_data = None
            for i, row in enumerate(value):
                for j, is_active in enumerate(row):
                    if is_active:
                        if not first_wi_data:
                            first_wi_data = self._clients_data[i][j]['wi_data']
                    self._clients_data[i][j]['active'] = is_active
            if first_wi_data:
                self.wi.update_values(first_wi_data)
            self._check_conflicting_data()

        self.wi.add_grid('active frames', rows=self.rows, cols=self.cols, value=True, callback=set_active_client)

        def toggle_field(c, value):
            c.frame.field_name = value

        options = defaultframe.fieldTypes.keys()
        self.wi.add_select('field', options=options, callback=self._callback(toggle_field))

        self.wi.make_tab('Control Panel', ['active frames'], ['field'])

    def _make_camera_tab(self, defaultframe):

        def set_drawing_style(c, value):
            c.set_drawing_style(value)
        self.wi.add_select('drawing style', options=['2D', '3D'], callback=self._callback(set_drawing_style))

        def set_camera_sync(value):
            active_clients = self._active_clients
            if len(active_clients) < 2:
                return

            ac0 = active_clients[0]
            if value:
                for ac in active_clients[1:]:
                    ac0.sync_cameras(ac)
            else:
                for ac in active_clients[1:]:
                    ac0.unsync_cameras(ac)

        self.wi.add_toggle('camera sync', callback=set_camera_sync, value=False)

        def set_depth_x(value):
            c.set_plane('x', value)

        def set_depth_y(value):
            c.draw()

        def set_depth_z(value):
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
        def toggle_bounding_box(c, value):
            c.frame.bounding_box_on = value

        def toggle_cell_borders(c, value):
            c.frame.cell_borders_on = value

        def toggle_cell_glyphs(c, value):
            c.frame.cell_glyphs_on = value

        def toggle_cells(c, value):
            c.frame.cells_on = value

        def toggle_cluster_borders(c, value):
            c.frame.cluster_borders_on = value

        def toggle_fpp_links(c, value):
            c.frame.fpp_links_on = value

        def toggle_lattice_axes_labels(c, value):
            c.frame.lattice_axes_labels_on = value

        def toggle_lattice_axes(c, value):
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
        frame_option_widget_names = [field for (field, _, _) in frame_options]
        options1, options2 = split_half(frame_option_widget_names)

        self.wi.make_tab('Visualization', options1, options2)

    def _make_colors_tab(self, defaultframe):
        # callback
        def set_color(field_name, value):
            for i, row in enumerate(self._clients_data):
                for j, data in enumerate(row):
                    c = data['client']
                    ###
                    n = int(field_name[-1])  # color number
                    c.frame.colormap[n] = Color.from_str_rgb(value)
                    c.frame.config.setSetting('TypeColorMap', c.frame.colormap)
                    ###
                    self._update_client_data(i, j)
                    c.frame.on_cell_type_color()
                    c.draw()
            self._check_conflicting_data()

        colorpicker_names_values = [(f'cell color {k}', str(v)) for (k, v) in defaultframe.colormap.items()]
        for (name, value) in colorpicker_names_values:
            self.wi.add_color(name, value=value, callback=set_color)

        colorpicker_names = [name for (name, value) in colorpicker_names_values]
        colors1, colors2 = split_half(colorpicker_names)

        self.wi.make_tab('Cell Colors', colors1, colors2)

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
            
        # debug
        # print('actives', len(active_client_datas))
        
        if len(active_client_datas) > 0:
            self.wi.set_disabled(False)
            for key, val in active_client_datas[0].items():
                if key in ['active frames', 'field']:
                    continue
                conflict = False
                for cd in active_client_datas:
                    if val != cd[key]:
                        conflict = True
                marked = '*' in self.wi.widgets[key].description

                # debug
                # if key in ['bounding box', 'cell borders']:
                #     print('conflict', conflict, 'marked', marked, key)

                if conflict and not marked:
                    self.wi.widgets[key].description = key+'*'
                elif not conflict and marked:
                    self.wi.widgets[key].description = self.wi.widgets[key].description.replace('*', '')
        else:
            self.wi.set_disabled(True)
