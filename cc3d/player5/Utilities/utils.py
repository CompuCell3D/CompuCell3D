# From core; do not remove unless you're looking for trouble, or making Player-specific implementations!
from cc3d.core.GraphicsUtils.utils import *


def qcolor_to_rgba(qcolor: object) -> tuple:
    """
    Converts qcolor to rgba tuple

    :param qcolor: {QColor}
    :return: {tuple (int, int, int, int)} rgba
    """

    return (qcolor.red(), qcolor.green(), qcolor.blue(), qcolor.alpha())
