"""
Class that specifies drawing scene properties/settings
"""


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
