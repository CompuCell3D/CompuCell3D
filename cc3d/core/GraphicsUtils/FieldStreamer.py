"""
Defines features for serializing/deserializing VTK visualization data

Permanent implementation resides in CompuCell3D/core/pyinterface/PlayerPythonNew

Porting is included here to prevent requiring custom builds to be built by friends who aren't well set up to do so
"""

import numpy
from typing import Dict, List, Optional, Tuple, Union, Any

from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonDataModelPython import vtkStructuredPoints

from cc3d.cpp.CompuCell import Dim3D
from cc3d.core.GraphicsUtils.utils import extract_address_int_from_vtk_object, recover_vtk_object_from_address_int

from cc3d.core.GraphicsUtils.prototypes.FieldWriterCML import FieldTypeCML, FieldWriterCML

try:
    from cc3d.cpp import PlayerPython
    from cc3d.cpp.PlayerPython import FieldStreamer as FieldStreamerCpp
    from cc3d.cpp.PlayerPython import FieldStreamerData as FieldStreamerDataCpp
except ImportError:
    FieldStreamerCpp = None
    FieldStreamerDataCpp = None


class FieldStreamerData:
    """
    Data container for :class:`FieldStreamer` instances. Safe for serialization.
    """

    def __init__(self):

        self.cell_field_names: List[str] = []
        self.conc_field_names: List[str] = []
        self.scalar_field_names: List[str] = []
        self.scalar_field_cell_level_names: List[str] = []
        self.vector_field_names: List[str] = []
        self.vector_field_cell_level_names: List[str] = []
        self.field_dim: Optional[Tuple[int, int, int]] = None
        self.data: Optional[Dict[str, numpy.ndarray]] = None

    def getFieldNames(self) -> List[str]:
        """All field names available in the container"""

        r = self.cell_field_names.copy()
        r.extend(self.conc_field_names.copy())
        r.extend(self.scalar_field_names.copy())
        r.extend(self.scalar_field_cell_level_names.copy())
        r.extend(self.vector_field_names.copy())
        r.extend(self.vector_field_cell_level_names.copy())
        return r

    @staticmethod
    def summarize(field_writer: FieldWriterCML):
        """Generate a container from a field writer"""

        obj = FieldStreamerData()

        field_dim: Dim3D = field_writer.getFieldDim()
        obj.field_dim = (field_dim.x, field_dim.y, field_dim.z)

        enum_flex = lambda e: e.value if hasattr(e, 'value') else e

        for i in range(field_writer.numFields()):
            field_name = field_writer.getFieldName(i)
            field_type = field_writer.getFieldType(i)

            if enum_flex(field_type) == enum_flex(FieldTypeCML.CellField):
                obj.cell_field_names.append(field_name)
            elif enum_flex(field_type) == enum_flex(FieldTypeCML.ConField):
                obj.conc_field_names.append(field_name)
            elif enum_flex(field_type) == enum_flex(FieldTypeCML.ScalarField):
                obj.scalar_field_names.append(field_name)
            elif enum_flex(field_type) == enum_flex(FieldTypeCML.ScalarFieldCellLevel):
                obj.scalar_field_cell_level_names.append(field_name)
            elif enum_flex(field_type) == enum_flex(FieldTypeCML.VectorField):
                obj.vector_field_names.append(field_name)
            elif enum_flex(field_type) == enum_flex(FieldTypeCML.VectorFieldCellLevel):
                obj.vector_field_cell_level_names.append(field_name)

        return obj

    @property
    def fieldDim(self):
        return Dim3D(*self.field_dim)


class FieldStreamer:
    """
    Support class for serializing/deserializing VTK data using a :class:`FieldWriterCML` instance.

    Usage example,

    .. code-block:: python

        import pickle
        fw: FieldWriterCML
        b = pickle.dumps(FieldStreamer(field_writer=fw))
        fs: FieldStreamer = pickle.loads(b)
        fe: FieldExtractorCML = FieldStreamer.loade(fs.data)

    """

    def __init__(self, data: FieldStreamerData = None):

        self.data = data

        self._points = None

    @property
    def points(self) -> vtkStructuredPoints:
        """Structured point data, derived from all available array data"""

        if self._points is None:
            self._points = self._loadp(self.data, False)
        return self._points

    @staticmethod
    def dump(field_writer: FieldWriterCML) -> FieldStreamerData:
        """Dump field writer data to a data container"""

        fsd = FieldStreamerData.summarize(field_writer)

        fsd.data = {}
        for name in fsd.getFieldNames():
            field_data = recover_vtk_object_from_address_int(field_writer.getArrayAddr(name))
            conv_data = numpy_support.vtk_to_numpy(field_data)
            if conv_data is None:
                raise RuntimeError('Data could not be dumped:', name)
            fsd.data[name] = conv_data

        return fsd

    def getPointsAddr(self) -> int:

        if self.points is None:
            return 0
        return extract_address_int_from_vtk_object(self.points)

    def getFieldDim(self) -> Dim3D:

        if self.data is None:
            return Dim3D()
        return self.data.fieldDim

    @staticmethod
    def _loadp(data: FieldStreamerData, copy: bool = True) -> vtkStructuredPoints:
        """
        Load structured points from a data container

        :param data: data container
        :type data: FieldStreamerData
        :param copy: flag to perform deep copy of array data
        :type copy: bool
        :return: structured points
        :rtype: vtkStructuredPoints
        """

        if data.data is None:
            raise ValueError('Data is empty')

        result = vtkStructuredPoints()
        for name in data.getFieldNames():
            vtk_array = numpy_support.numpy_to_vtk(data.data[name], int(copy))
            vtk_array.SetName(name)
            result.GetPointData().AddArray(vtk_array)

        return result


if FieldStreamerCpp is not None:

    class FieldStreamerDataPy(FieldStreamerDataCpp):

        def __reduce__(self) -> Union[str, Tuple[Any, ...]]:

            return _fieldstreamerdatapy_serial, (list(self.cellFieldNames),
                                                 list(self.concFieldNames),
                                                 list(self.scalarFieldNames),
                                                 list(self.scalarFieldCellLevelNames),
                                                 list(self.vectorFieldNames),
                                                 list(self.vectorFieldCellLevelNames),
                                                 self.fieldDim,
                                                 self.data)

        @staticmethod
        def from_base(fsd_cpp: FieldStreamerCpp):
            return _fieldstreamerdatapy_serial(list(fsd_cpp.cellFieldNames),
                                               list(fsd_cpp.concFieldNames),
                                               list(fsd_cpp.scalarFieldNames),
                                               list(fsd_cpp.scalarFieldCellLevelNames),
                                               list(fsd_cpp.vectorFieldNames),
                                               list(fsd_cpp.vectorFieldCellLevelNames),
                                               fsd_cpp.fieldDim,
                                               fsd_cpp.data)

        @staticmethod
        def to_base(fsd):
            fsd: FieldStreamerDataPy
            fsd_cpp = FieldStreamerDataCpp()
            fsd.cellFieldNames = fsd_cpp.cellFieldNames
            fsd.concFieldNames = fsd_cpp.concFieldNames
            fsd.scalarFieldNames = fsd_cpp.scalarFieldNames
            fsd.scalarFieldCellLevelNames = fsd_cpp.scalarFieldCellLevelNames
            fsd.vectorFieldNames = fsd_cpp.vectorFieldNames
            fsd.vectorFieldCellLevelNames = fsd_cpp.vectorFieldCellLevelNames
            fsd.fieldDim = fsd_cpp.fieldDim
            fsd.data = fsd_cpp.data
            return fsd_cpp

    FieldStreamerData = FieldStreamerDataPy

    def _fieldstreamerdatapy_serial(cellFieldNames: List[str],
                                    concFieldNames: List[str],
                                    scalarFieldNames: List[str],
                                    scalarFieldCellLevelNames: List[str],
                                    vectorFieldNames: List[str],
                                    vectorFieldCellLevelNames: List[str],
                                    fieldDim: Dim3D,
                                    data):
        r = FieldStreamerDataPy()
        [r.cellFieldNames.push_back(s) for s in cellFieldNames]
        [r.concFieldNames.push_back(s) for s in concFieldNames]
        [r.scalarFieldNames.push_back(s) for s in scalarFieldNames]
        [r.scalarFieldCellLevelNames.push_back(s) for s in scalarFieldCellLevelNames]
        [r.vectorFieldNames.push_back(s) for s in vectorFieldNames]
        [r.vectorFieldCellLevelNames.push_back(s) for s in vectorFieldCellLevelNames]
        r.fieldDim = fieldDim
        r.data = data
        return r

    class FieldStreamerPy(FieldStreamerCpp):

        def __init__(self, data: FieldStreamerData = None):

            super().__init__()

            if data is not None:
                self.loadData(data)

        @classmethod
        def from_datapy(cls, data: FieldStreamerDataPy):
            return cls(FieldStreamerDataPy.to_base(data))

        def __reduce__(self) -> Union[str, Tuple[Any, ...]]:
            return FieldStreamerPy.from_datapy, (FieldStreamerDataPy.from_base(self.getData()),)

        @property
        def points(self) -> vtkStructuredPoints:
            """Structured point data, derived from all available array data"""
            return vtkStructuredPoints(hex(self.getPointsAddr()))

        @staticmethod
        def dump(field_writer: FieldWriterCML) -> FieldStreamerData:
            return FieldStreamerDataPy.from_base(FieldStreamerCpp.dump(field_writer))

    FieldStreamer = FieldStreamerPy
