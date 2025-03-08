from enum import Enum

from dataclasses import dataclass
from typing import Literal, Union


@dataclass
class FieldProperties:
    field_type: Literal["CellField", "ConField", "ScalarField", "ScalarFieldCellLevel", "VectorField", "VectorFieldCellLevel", "CustomVis"]
    precision_type: Literal["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64", "float32", "float64", "float128"]
    field_type_id: int = None
    field_name: str = None

def get_field_type(field_type_obj: Union[str, None, FieldProperties]):
    if not field_type_obj:
        return None
    if isinstance(field_type_obj, FieldProperties):
        return field_type_obj.field_type
    return field_type_obj

def get_field_precision_type(field_type_obj: Union[str, None, FieldProperties]):
    if not field_type_obj:
        return None
    if isinstance(field_type_obj, FieldProperties):
        return field_type_obj.precision_type
    return None


(CELL_FIELD, CON_FIELD, SCALAR_FIELD, SCALAR_FIELD_CELL_LEVEL, VECTOR_FIELD, VECTOR_FIELD_CELL_LEVEL, SCALAR_FIELD_NPY,
 VECTOR_FIELD_NPY, CUSTOM_FIELD, SHARED_SCALAR_NUMPY_FIELD, SHARED_VECTOR_NUMPY_FIELD) = range(0, 11)
(LEGACY_FORMAT, CSV_FORMAT) = range(0, 2)

(STOP_STATE, RUN_STATE, STEP_STATE, PAUSE_STATE) = list(range(0, 4))

GRAPHICS_WINDOW_LABEL, PLOT_WINDOW_LABEL, STEERING_PANEL_LABEL, MESSAGE_WINDOW_LABEL = (
    'Graphics', 'Plot', 'Steering_Panel', 'Message')

FIELD_NUMBER_TO_FIELD_TYPE_MAP = {
    CELL_FIELD: "CellField",
    CON_FIELD: "ConField",
    SCALAR_FIELD: "ScalarField",
    SCALAR_FIELD_CELL_LEVEL: "ScalarFieldCellLevel",
    VECTOR_FIELD: "VectorField",
    VECTOR_FIELD_CELL_LEVEL: "VectorFieldCellLevel",
    SCALAR_FIELD_NPY: "ScalarField",
    VECTOR_FIELD_NPY: "VectorField",
    CUSTOM_FIELD: "CustomVis",
    SHARED_SCALAR_NUMPY_FIELD: "ConField",
    SHARED_VECTOR_NUMPY_FIELD: "VectorField",
}


# Simulation types
class SimType(Enum):
    # "regular" run
    AUTOMATED = 'Auto'
    # general threaded run; this is the signal for injecting a SimulationThread into the core
    THREADED = 'Thread'
    # service mode; special case of THREADED for future dev support
    SERVICE = 'Service'
