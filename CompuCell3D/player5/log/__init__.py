import logging
import inspect
_LOG_FORMAT = '%(asctime)s : %(name)s : %(levelname)s : LN # = %(lineno)s : %(message)s'
def setup_logging(logpath=None, level=logging.INFO, disable_qt_log=True, ignore=[]):
    """Configure RAM logging output.

    :param str logpath: Log path.
    :param int level: Minimum log level to log
    :param bool disable_qt_log: When True (default), disable PyQt4 loggers.
    :param list ignore: Loggers to suppress from the stream output (they will
        still be logged to SQLite).

    """
    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        # f_handler = SQLiteHandler(logpath, level=level)

        logger.addHandler(handler)
        # logger.addHandler(f_handler)

    # if disable_qt_log:
    #     logging.getLogger("PyQt4").propagate = False

def get_logger(name=''):
    frm = inspect.stack()[1]
    mod = inspect.getmodule(frm[0])
    # print 'frm=',frm
    # print 'mod=', mod

    if not name:
        name = mod.__name__
    # print
    return logging.getLogger(name)


# logger = logging.getLogger('player5')
# logger.setLevel(logging.DEBUG)
# # create console handler and set level to debug
# ch = logging.StreamHandler()
# ch.setLevel(logging.DEBUG)
# # create formatter
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# # add formatter to ch
# ch.setFormatter(formatter)
# # add ch to logger
# logger.addHandler(ch)
