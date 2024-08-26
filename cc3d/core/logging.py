from cc3d.cpp import CompuCell
from contextlib import ContextDecorator
import inspect
import warnings


forward_warnings = True


def get_logger() -> CompuCell.CC3DLogger:
    return CompuCell.CC3DLogger.get()


def _log_py(_frame, log_level=CompuCell.LOG_CURRENT, msg=''):
    get_logger().logf(log_level, msg, _frame.f_code.co_name, inspect.getfile(_frame), _frame.f_lineno)

    if forward_warnings and log_level <= CompuCell.LOG_WARNING:
        warnings.warn(msg)


def log_py(log_level=CompuCell.LOG_CURRENT, msg='', levels_back=0):
    frame = inspect.currentframe()
    for _ in range(levels_back):
        frame = frame.f_back
    _log_py(frame, log_level, msg)


class LoggedContext(ContextDecorator):

    def __init__(self,
                 log_level=CompuCell.LOG_TRACE,
                 msg=''):

        self._level = log_level
        self._msg = msg
        self._frame = inspect.currentframe().f_back

    def __enter__(self):
        self.log()
        return self

    def __exit__(self, *exc):
        self.log()
        return False

    def log(self, level=None, msg: str = None):
        _log_py(self._frame,
                self._level if level is None else level,
                self._msg if msg is None else msg)
