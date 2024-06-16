"""
Defines comm protocols for CC3DPyGraphicsFrame
"""

from enum import Enum
from multiprocessing.connection import Connection
from typing import Any, Optional, Tuple, Union


class CC3DPyGraphicsFrameConnectionMsgType(Enum):
    """Labels for annotating frame connection messages"""

    REQUESTING = 0
    SERVICING = 1


class CC3DPyGraphicsFrameConnectionMsg:
    """
    Basic message container for message passing between a client processes and a rendering process running
    a graphics frame.

    The class attribute :attr:`method` is called on the object connected on the other side of the pipe, with
    positional and keyword arguments as specified in the call to :meth:`request`.
    """

    method: str = None

    def __init__(self, msg_type: CC3DPyGraphicsFrameConnectionMsgType, result, *args, **kwargs):
        """

        :param msg_type: message type
        :type msg_type: CC3DPyGraphicsFrameConnectionMsgType
        :param result: message result
        :type result: Any
        :param args: message arguments
        :param kwargs: message keyword arguments
        """

        self.msg_type = msg_type
        self.result = result
        self.args = args
        self.kwargs = kwargs

    @classmethod
    def request(cls, conn: Connection, return_result: bool, *args, **kwargs):
        """Request that the object on the other side of the pipe execute the associated message"""

        msg = cls(CC3DPyGraphicsFrameConnectionMsgType.REQUESTING, None, *args, **kwargs)
        conn.send(msg)
        if return_result:
            msg = conn.recv()
            if not isinstance(msg, CC3DPyGraphicsFrameConnectionMsg):
                raise RuntimeError
            return msg.result

    # def service(self, obj, conn: Connection = None):
    def service(self, *args, **kwargs):
        """Instructions for executing the request by the object on the other side of the pipe"""

        obj = kwargs.get('obj')
        if obj is None:
            obj = args[0]

        conn: Optional[Connection] = kwargs.get('conn')
        if conn is None and len(args) == 2:
            conn = args[1]

        self.result = getattr(obj, self.method)(*self.args, **self.kwargs)
        if conn is not None:
            self.msg_type = CC3DPyGraphicsFrameConnectionMsgType.SERVICING
            conn.send(self)


class FrameControlMessage:
    """
    Basic message container for message passing between a graphics frame process and its controller.

    The method :meth:`process` describes instructions to be executed by the frame.
    The method :meth:`control` describes instructions to be executed by the controller after issuing the message
    to the frame.
    """

    def __init__(self, msg_id: int = 0):

        self.id = msg_id
        """Unique message id. The id of a message in the client and rendering processes is the same."""

    def process(self, proc):
        """
        Executed by the graphics frame process

        :param proc: graphics frame process
        :type proc: cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame.CC3DPyGraphicsFrameProcess
        :return: None
        """

    def control(self, control):
        """
        Executed by the graphics frame process controller

        :param control: graphics frame process controller
        :type control: cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame.CC3DPyGraphicsFrameControlInterface
        :return: evaluation return result
        :rtype: Any
        """

    def control_self_blocking(self, control):
        """
        Instruction for control to block until it finds its image in the output queue

        :param control: graphics frame process controller
        :type control: cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame.CC3DPyGraphicsFrameControlInterface
        :return: message
        :rtype: FrameControlMessage
        """
        while True:
            msg: FrameControlMessage = control.frame_proc.queue_output.get()
            if msg.id == self.id:
                return msg
            control.frame_proc.queue_output.put(msg)


class MsgNoReturn(CC3DPyGraphicsFrameConnectionMsg):
    """A message with no return object required"""

    def service(self, obj, conn: Connection = None):
        """Service the message without a pipe connection for returning data"""
        super().service(obj)


class MsgWithReturn(CC3DPyGraphicsFrameConnectionMsg):
    """A message with a return object required"""

    def service(self, obj, conn: Connection):
        """Service the message with a pipe connection for returning data"""
        super().service(obj, conn)


class MsgGetSetting(MsgWithReturn):
    """Message to get a configuration setting"""

    method = '_service_get_config'


class MsgGetConcFieldNames(MsgWithReturn):
    """Message to stream all current concentration field names"""

    method = '_service_get_concentration_field_names'


class MsgGetFieldsToCreate(MsgWithReturn):
    """Message to stream all current names of fields to create"""

    method = '_service_get_fields_to_create'


class MsgGetStreamedData(MsgWithReturn):
    """Message to stream field data"""

    method = '_service_get_streamed_data'


class MsgGetBasicSimData(MsgWithReturn):
    """Message to stream basic simulation data"""

    method = '_service_get_bsd'


class MsgGetLatticeTypeStr(MsgWithReturn):
    """Message to stream the name of the lattice type"""

    method = '_service_get_lattice_type_str'


class MsgInitFieldTypesFrameInterface(MsgWithReturn):
    """Message to request field names and types as a dictionary"""

    method = '_service_init_field_types'


class MsgShutdownFrameInterface(MsgNoReturn):
    """Message to request that a graphics frame interface shutdown"""

    method = 'close'


class ControlMessageDraw(FrameControlMessage):
    """
    Message to draw an update.

    If the message is blocking, then the call will block until execution of the message is complete.
    Otherwise, the frame will only execute the last request to draw when updating and disregard all other requests.
    """

    def __init__(self, blocking: bool, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.blocking = blocking

    def process(self, proc):
        """
        Execute draw and return self to output queue if blocking

        :param proc: graphics frame process
        :type proc: cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame.CC3DPyGraphicsFrameProcess
        :return: None
        """
        proc.frame.draw()
        if self.blocking:
            proc.queue_output.put(self)

    def control(self, control):
        """
        Wait for image of self to be returned in output queue if blocking

        :param control: graphics frame process controller
        :type control: cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame.CC3DPyGraphicsFrameControlInterface
        :return: None
        """
        if self.blocking:
            self.control_self_blocking(control)


class ControlMessageShutdown(FrameControlMessage):
    """Message to close the graphics frame process"""

    def process(self, proc):
        """
        Shut down the frame

        :param proc: graphics frame process
        :type proc: cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame.CC3DPyGraphicsFrameProcess
        :return: None
        """
        proc.frame.shutdown()


class ControlMessageGetAttr(FrameControlMessage):
    """Message to get the value of an attribute on a frame"""

    def __init__(self, attr_name: str):

        super().__init__()

        self.attr_name = attr_name
        self.attr_val = None

    def process(self, proc):
        """
        Execute attribute value retrieval and return self to output queue

        :param proc: graphics frame process
        :type proc: cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame.CC3DPyGraphicsFrameProcess
        :return: None
        """
        try:
            self.attr_val = getattr(proc.frame, self.attr_name)
        except AttributeError:
            pass
        proc.queue_output.put(self)

    def control(self, control):
        """
        Wait for image of self to be returned in output queue

        :param control: graphics frame process controller
        :type control: cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame.CC3DPyGraphicsFrameControlInterface
        :return: Any
        """
        msg = self.control_self_blocking(control)
        return msg.attr_val


class ControlMessageSetAttr(FrameControlMessage):
    """Message to get the value of an attribute on a frame"""

    def __init__(self, attr_name: str, attr_val: Any, blocking: bool = False):

        super().__init__()

        self.attr_name = attr_name
        self.attr_val = attr_val
        self.blocking = blocking

    def process(self, proc):
        """
        Execute attribute value retrieval and return self to output queue

        :param proc: graphics frame process
        :type proc: cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame.CC3DPyGraphicsFrameProcess
        :return: None
        """
        try:
            setattr(proc.frame, self.attr_name, self.attr_val)
        except AttributeError:
            pass
        if self.blocking:
            proc.queue_output.put(self)

    def control(self, control):
        """
        Wait for image of self to be returned in output queue

        :param control: graphics frame process controller
        :type control: cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame.CC3DPyGraphicsFrameControlInterface
        :return: None
        """
        if self.blocking:
            self.control_self_blocking(control)


class ControlMessagePullMetadata(FrameControlMessage):

    def __init__(self, field_name: str):

        super().__init__()

        self.field_name = field_name

    def process(self, proc):
        """
        Update field metadata storage on a frame for a field

        :param proc: graphics frame process
        :type proc: cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame.CC3DPyGraphicsFrameProcess
        :return: None
        """
        proc.frame.pull_metadata(field_name=self.field_name)


class ControlMessageSetFieldName(FrameControlMessage):
    """Message to set the name of the rendering target field"""

    def __init__(self, field_name: str):

        super().__init__()

        self.field_name = field_name

    def process(self, proc):
        """
        Set the field name on the frame

        :param proc: graphics frame process
        :type proc: cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame.CC3DPyGraphicsFrameProcess
        :return: None
        """
        proc.frame.set_field_name(self.field_name)


class ControlMessageSetPlanePos(FrameControlMessage):
    """Message to set the plane and position"""

    def __init__(self, plane: str, pos: int):

        super().__init__()

        self.plane = plane
        self.pos = pos

    def process(self, proc):
        """
        Set the plane and position on the frame

        :param proc: graphics frame process
        :type proc: cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame.CC3DPyGraphicsFrameProcess
        :return: None
        """
        proc.frame.set_plane(self.plane, self.pos)


class ControlMessageSetDrawingStyle(FrameControlMessage):
    """Message to set the drawing style"""

    def __init__(self, _style: str):

        if _style not in ['2D', '3D']:
            raise RuntimeError('Valid styles are \'2D\' and \'3D\'')

        super().__init__()

        self.style = _style

    def process(self, proc):
        """
        Set the drawing style on the frame

        :param proc: graphics frame process
        :type proc: cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame.CC3DPyGraphicsFrameProcess
        :return: None
        """
        proc.frame.set_drawing_style(self.style)


class ControlMessageGetNPImageData(FrameControlMessage):
    """Message to export image data as a numpy array"""

    def __init__(self,
                 scale: Union[int, Tuple[int, int]] = None,
                 transparent_background: bool = False):

        super().__init__()

        self.scale = scale
        self.transparent_background = transparent_background
        self.np_data = None

    def process(self, proc):
        """
        Generate and return numpy data of an image of the rendered scene in the frame

        :param proc: graphics frame process
        :type proc: cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame.CC3DPyGraphicsFrameProcess
        :return: None
        """
        self.np_data = proc.frame.np_img_data(scale=self.scale, transparent_background=self.transparent_background)
        proc.queue_output.put(self)

    def control(self, control):
        """
        Issue request with blocking call to return data

        :param control: graphics frame process controller
        :type control: cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame.CC3DPyGraphicsFrameControlInterface
        :return: numpy data
        :rtype: ndarray
        """
        msg: ControlMessageGetNPImageData = self.control_self_blocking(control)
        return msg.np_data


class ControlMessageSaveImage(FrameControlMessage):
    """Message to save to file"""

    def __init__(self,
                 file_path: str,
                 blocking: bool = False,
                 scale: Union[int, Tuple[int, int]] = None,
                 transparent_background: bool = False):

        super().__init__()

        self.file_path = file_path
        self.blocking = blocking
        self.scale = scale
        self.transparent_background = transparent_background

    def process(self, proc):
        """
        Save image to file

        :param proc: graphics frame process
        :type proc: cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame.CC3DPyGraphicsFrameProcess
        :return: None
        """
        proc.frame.save_img(file_path=self.file_path,
                            scale=self.scale,
                            transparent_background=self.transparent_background)
        if self.blocking:
            proc.queue_output.put(self)

    def control(self, control):
        """
        Do blocking if requested

        :param control: graphics frame process controller
        :type control: cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame.CC3DPyGraphicsFrameControlInterface
        :return: None
        """
        if self.blocking:
            self.control_self_blocking(control)


class ControlMessageGetScreenshotData(FrameControlMessage):
    """Message to get serialized screenshot data"""

    def __init__(self):

        super().__init__()

        self.screenshot_data = None

    def process(self, proc):
        """Get screenshot data"""
        self.screenshot_data = proc.frame.get_screenshot_data().to_json()
        proc.queue_output.put(self)

    def control(self, control):
        """
        Issue request with blocking call to return data

        :param control: graphics frame process controller
        :type control: cc3d.core.GraphicsUtils.CC3DPyGraphicsFrame.CC3DPyGraphicsFrameControlInterface
        :return: screenshot data
        :rtype: dict
        """
        msg: ControlMessageGetScreenshotData = self.control_self_blocking(control)
        return msg.screenshot_data
