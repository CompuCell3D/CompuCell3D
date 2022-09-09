from enum import Enum

(CELL_FIELD, CON_FIELD, SCALAR_FIELD, SCALAR_FIELD_CELL_LEVEL, VECTOR_FIELD, VECTOR_FIELD_CELL_LEVEL, SCALAR_FIELD_NPY,
 VECTOR_FIELD_NPY, CUSTOM_FIELD) = range(0, 9)
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
    CUSTOM_FIELD: "CustomVis"
}


# Simulation types
class SimType(Enum):
    # "regular" run
    AUTOMATED = 'Auto'
    # general threaded run; this is the signal for injecting a SimulationThread into the core
    THREADED = 'Thread'
    # service mode; special case of THREADED for future dev support
    SERVICE = 'Service'
