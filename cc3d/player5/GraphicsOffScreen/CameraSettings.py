class CameraSettings:
    def __init__(self):
        self.position = [0., 0., 0.]
        self.focalPoint = [0., 0., 0.]
        self.viewUp = [0., 0., 0.]
        self.viewPlaneNormal = [0., 0., 0.]
        self.clippingRange = [0., 0.]
        self.distance = 0.0
        self.viewAngle = 0.0
        self.parallelScale = 1.0

    def __str__(self):
        return "position=" + str(self.position) + '\n' + ' focalPoint=' + str(
            self.focalPoint) + '\n' + ' viewUp=' + str(self.viewUp) + '\n' + ' clippingRange=' + str(
            self.clippingRange) + '\n' + ' distance=' + str(self.distance) + '\n'
