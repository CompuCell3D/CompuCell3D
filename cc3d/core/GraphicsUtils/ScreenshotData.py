"""
Class that specifies drawing scene properties/settings
"""
import json
from collections import OrderedDict


class ScreenshotData(object):
    def __init__(self):
        self.screenshotName = ""
        self.screenshotCoreName = ""
        self.spaceDimension = "2D"
        self.projection = "xy"

        # this is a tuple where first element is field name (as displayed in the field list in the player5)
        # and the second one is plot type (e.g. CellField, Confield, Vector Field)
        self.plotData = ("Cell_Field",
                         "CellField")
        # self.out_dir_core_name = ""
        self.projectionPosition = 0
        self.screenshotGraphicsWidget = None
        # self.originalCameraObj=None
        # those are the values used to handle 3D screenshots

        self.clippingRange = None
        self.focalPoint = None
        self.position = None
        self.viewUp = None
        self.cell_borders_on = None
        self.cells_on = None
        self.cluster_borders_on = None
        self.cell_glyphs_on = None
        self.fpp_links_on = None
        self.bounding_box_on = None
        self.lattice_axes_on = None
        self.lattice_axes_labels_on = None
        self.invisible_types = None
        self.win_width = 299
        self.win_height = 299
        self.cell_shell_optimization = None

        self.metadata = {}

    def extractCameraInfo(self, _camera):
        self.clippingRange = _camera.GetClippingRange()
        self.focalPoint = _camera.GetFocalPoint()
        self.position = _camera.GetPosition()
        self.viewUp = _camera.GetViewUp()

    def extractCameraInfoFromList(self, _cameraSettings):
        self.clippingRange = _cameraSettings[0:2]
        self.focalPoint = _cameraSettings[2:5]
        self.position = _cameraSettings[5:8]
        self.viewUp = _cameraSettings[8:11]

    def prepareCamera(self):

        if self.clippingRange and self.focalPoint and self.position and self.viewUp:
            cam = self.screenshotGraphicsWidget.get_camera()
            cam.SetClippingRange(self.clippingRange)
            cam.SetFocalPoint(self.focalPoint)
            cam.SetPosition(self.position)
            cam.SetViewUp(self.viewUp)

    def compareCameras(self, _camera):

        if self.clippingRange != _camera.GetClippingRange():
            return False
        if self.focalPoint != _camera.GetFocalPoint():
            return False
        if self.position != _camera.GetPosition():
            return False
        if self.viewUp != _camera.GetViewUp():
            return False
        return True

    def compareExistingCameraToNewCameraSettings(self, _cameraSettings):
        if self.clippingRange[0] != _cameraSettings[0] or self.clippingRange[1] != _cameraSettings[1]:
            return False
        if self.focalPoint[0] != _cameraSettings[2] or self.focalPoint[1] != _cameraSettings[3] or self.focalPoint[3] != \
                _cameraSettings[4]:
            return False
        if self.position[0] != _cameraSettings[5] or self.position[1] != _cameraSettings[6] or self.position[3] != \
                _cameraSettings[7]:
            return False
        if self.viewUp[0] != _cameraSettings[8] or self.viewUp[1] != _cameraSettings[9] or self.viewUp[3] != \
                _cameraSettings[10]:
            return False

    def to_json(self):
        """Generates a JSON-compatible data structure"""

        scr_elem = OrderedDict()
        scr_elem['Plot'] = {'PlotType': str(self.plotData[1]), 'PlotName': str(self.plotData[0])}

        if self.spaceDimension == '2D':
            scr_elem['Dimension'] = '2D'

            scr_elem['Projection'] = {'ProjectionPlane': self.projection,
                                      'ProjectionPosition': int(self.projectionPosition)}

        if self.spaceDimension == '3D':
            scr_elem['Dimension'] = '3D'
            scr_elem['Projection'] = {'ProjectionPlane': None, 'ProjectionPosition': None}

        scr_elem['CameraClippingRange'] = {
            'Min': str(self.clippingRange[0]),
            'Max': str(self.clippingRange[1])
        }

        scr_elem['CameraFocalPoint'] = {
            'x': str(self.focalPoint[0]),
            'y': str(self.focalPoint[1]),
            'z': str(self.focalPoint[2])
        }

        scr_elem['CameraPosition'] = {
            'x': str(self.position[0]),
            'y': str(self.position[1]),
            'z': str(self.position[2])
        }

        scr_elem['CameraViewUp'] = {
            'x': str(self.viewUp[0]),
            'y': str(self.viewUp[1]),
            'z': str(self.viewUp[2])
        }

        scr_elem['Size'] = {
            'Width': int(self.win_width),
            'Height': int(self.win_height)
        }

        scr_elem['CellBorders'] = bool(self.cell_borders_on)
        scr_elem['Cells'] = bool(self.cells_on)
        scr_elem['ClusterBorders'] = bool(self.cluster_borders_on)
        scr_elem['CellGlyphs'] = bool(self.cell_glyphs_on)
        scr_elem['FPPLinks'] = bool(self.fpp_links_on)
        scr_elem['BoundingBox'] = bool(self.bounding_box_on)
        scr_elem['LatticeAxes'] = bool(self.lattice_axes_on)
        scr_elem['LatticeAxesLabels'] = bool(self.lattice_axes_labels_on)

        if self.invisible_types is None:
            self.invisible_types = []
        scr_elem['TypesInvisible'] = self.invisible_types
        scr_elem["CellShellOptimization"] = self.cell_shell_optimization

        scr_elem['metadata'] = self.metadata
        scr_elem['screenshotName'] = self.screenshotName

        return scr_elem

    def to_json_simulate_file_readout(self):
        """
        Generates a JSON-compatible data structure by first writing json dict to string then loading it.
        This ensures that keys are strings and not floats, integers, etc...

        :return:
        """
        return json.loads(json.dumps(self.to_json()))

    @classmethod
    def from_json(cls, _data, scr_name: str = None):
        """Constructs an instance from a JSON-compatible data structure"""

        scr_data = cls()
        scr_data.screenshotName = scr_name if scr_name is not None else _data['Plot']['PlotName']

        scr_data.plotData = tuple(
            [str(x) for x in (_data['Plot']['PlotName'], _data['Plot']['PlotType'])])
        scr_data.spaceDimension = str(_data['Dimension'])
        try:
            scr_data.projection = str(_data['Projection']['ProjectionPlane'])
            scr_data.projectionPosition = _data['Projection']['ProjectionPosition']
        except KeyError:
            pass
        scr_data.win_width = _data['Size']['Width']
        scr_data.win_height = _data['Size']['Height']

        scr_data.cell_borders_on = _data['CellBorders']
        scr_data.cells_on = _data['Cells']
        scr_data.cluster_borders_on = _data['ClusterBorders']
        scr_data.cell_glyphs_on = _data['CellGlyphs']
        scr_data.fpp_links_on = _data['FPPLinks']
        scr_data.bounding_box_on = _data['BoundingBox']
        scr_data.lattice_axes_on = _data['LatticeAxes']
        scr_data.lattice_axes_labels_on = _data['LatticeAxesLabels']
        scr_data.invisible_types = _data['TypesInvisible']
        scr_data.cell_shell_optimization = _data.get('CellShellOptimization', False)

        cam_settings = []

        clipping_range_element = _data['CameraClippingRange']
        cam_settings.append(float(clipping_range_element['Min']))
        cam_settings.append(float(clipping_range_element['Max']))

        focal_point_element = _data['CameraFocalPoint']
        cam_settings.append(float(focal_point_element['x']))
        cam_settings.append(float(focal_point_element['y']))
        cam_settings.append(float(focal_point_element['z']))

        position_element = _data['CameraPosition']
        cam_settings.append(float(position_element['x']))
        cam_settings.append(float(position_element['y']))
        cam_settings.append(float(position_element['z']))

        view_up_element = _data['CameraViewUp']
        cam_settings.append(float(view_up_element['x']))
        cam_settings.append(float(view_up_element['y']))
        cam_settings.append(float(view_up_element['z']))

        scr_data.extractCameraInfoFromList(cam_settings)

        # getting rid of unicode in the keys
        metadata_dict = {}
        for k, v in list(_data['metadata'].items()):
            metadata_dict[str(k)] = v

        scr_data.metadata = metadata_dict

        # checking for extra metadata entries added
        # you may reset this list after bumping up the version of json file
        # todo fix - we will be permissive as far as DisplayMinMaxInfo
        # extra_metadata_keys = ['DisplayMinMaxInfo']
        extra_metadata_keys = []

        for key in extra_metadata_keys:
            if key not in list(metadata_dict.keys()):
                raise KeyError('Missing key in the metadata: {}'.format(key))

        return scr_data

    def __reduce__(self):
        return ScreenshotData.from_json, (self.to_json(), self.screenshotName)
