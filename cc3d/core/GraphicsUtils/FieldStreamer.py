"""
Defines features for serializing/deserializing VTK visualization data
"""

import numpy
from typing import Dict, List, Optional, Tuple

from vtkmodules.util import numpy_support
from vtkmodules.vtkCommonDataModelPython import vtkStructuredPoints

from cc3d.cpp.CompuCell import Dim3D
from cc3d.cpp.PlayerPython import FieldExtractorCML
from cc3d.core.GraphicsUtils.utils import extract_address_int_from_vtk_object

from .prototypes.FieldWriterCML import FieldTypeCML, FieldWriterCML


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

    @property
    def field_names(self) -> List[str]:
        """All field names available in the container"""

        r = self.cell_field_names.copy()
        r.extend(self.conc_field_names.copy())
        r.extend(self.scalar_field_names.copy())
        r.extend(self.scalar_field_cell_level_names.copy())
        r.extend(self.vector_field_names.copy())
        r.extend(self.vector_field_cell_level_names.copy())
        return r

    @property
    def names_by_type(self) -> Dict[FieldTypeCML, List[str]]:
        """Names of fields by field type"""

        return {FieldTypeCML.CellField: self.cell_field_names.copy(),
                FieldTypeCML.ConField: self.conc_field_names.copy(),
                FieldTypeCML.ScalarField: self.scalar_field_names.copy(),
                FieldTypeCML.ScalarFieldCellLevel: self.scalar_field_cell_level_names.copy(),
                FieldTypeCML.VectorField: self.vector_field_names.copy(),
                FieldTypeCML.VectorFieldCellLevel: self.vector_field_cell_level_names.copy()}

    @staticmethod
    def summarize(field_writer: FieldWriterCML):
        """Generate a container from a field writer"""

        obj = FieldStreamerData()

        field_dim: Dim3D = field_writer.sim.getPotts().getCellFieldG().getDim()
        obj.field_dim = field_dim.to_tuple()

        for i in range(len(field_writer.arrayNameVec)):
            field_name = field_writer.arrayNameVec[i]
            field_type = field_writer.arrayTypeVec[i]

            if field_type == FieldTypeCML.CellField:
                obj.cell_field_names.append(field_name)
            elif field_type == FieldTypeCML.ConField:
                obj.conc_field_names.append(field_name)
            elif field_type == FieldTypeCML.ScalarField:
                obj.scalar_field_names.append(field_name)
            elif field_type == FieldTypeCML.ScalarFieldCellLevel:
                obj.scalar_field_cell_level_names.append(field_name)
            elif field_type == FieldTypeCML.VectorField:
                obj.vector_field_names.append(field_name)
            elif field_type == FieldTypeCML.VectorFieldCellLevel:
                obj.vector_field_cell_level_names.append(field_name)

        return obj


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

    def __init__(self, data: FieldStreamerData = None, field_writer: FieldWriterCML = None):

        self.data = FieldStreamer.dump(field_writer) if field_writer is not None else data

        self._fields = None
        self._points = None

    @property
    def fields(self) -> dict:
        """Array data by field type and name"""

        if self._fields is None:
            self._fields = FieldStreamer.loadd(self.data, False)
        return self._fields

    @property
    def points(self) -> vtkStructuredPoints:
        """Structured point data, derived from all available array data"""

        if self._points is None:
            self._points = FieldStreamer.loadp(self.data, False)
        return self._points

    @staticmethod
    def dump(field_writer: FieldWriterCML) -> FieldStreamerData:
        """Dump field writer data to a data container"""

        if field_writer.latticeData is None:
            raise ValueError('Field writer is empty')

        fsd = FieldStreamerData.summarize(field_writer)

        fsd.data = {}
        for name in fsd.field_names:
            field_data = field_writer.latticeData.GetPointData().GetArray(name)
            conv_data = numpy_support.vtk_to_numpy(field_data)
            if conv_data is None:
                raise RuntimeError('Data could not be dumped:', name)
            fsd.data[name] = conv_data

        return fsd

    @staticmethod
    def loadp(data: FieldStreamerData, copy: bool = True) -> vtkStructuredPoints:
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
        for name in data.field_names:
            vtk_array = numpy_support.numpy_to_vtk(data.data[name], int(copy))
            vtk_array.SetName(name)
            result.GetPointData().AddArray(vtk_array)

        return result

    @staticmethod
    def loadd(data: FieldStreamerData, copy: bool = True) -> dict:
        """
        Load arrays from a data container

        :param data: data container
        :type data: FieldStreamerData
        :param copy: flag to perform deep copy of array data
        :type copy: bool
        :return: arrays by field type and name
        :rtype: dict
        """

        if data.data is None:
            raise ValueError('Data is empty')

        result = {}
        for field_type, field_type_names in data.names_by_type.items():
            result[field_type] = {}
            for field_name in field_type_names:
                vtk_array = numpy_support.numpy_to_vtk(data.data[field_name], copy)
                vtk_array.SetName(field_name)
                result[field_type][field_name] = vtk_array

        return result

    @staticmethod
    def loade(data: FieldStreamerData) -> FieldExtractorCML:
        """
        Load a field extractor from a data container

        :param data: data container
        :type data: FieldStreamerData
        :return: field extractor
        :rtype: FieldExtractorCML
        """

        if data.data is None:
            raise ValueError('Data is empty')

        return FieldStreamer.convert_p2e(data.field_dim, FieldStreamer.loadp(data))

    @staticmethod
    def convert_p2e(field_dim: Tuple[int, int, int], vtk_data: vtkStructuredPoints) -> FieldExtractorCML:
        """Convert structured points to a field extractor"""

        fe = FieldExtractorCML()
        fe.setFieldDim(Dim3D(*field_dim))
        fe.setSimulationData(extract_address_int_from_vtk_object(vtk_data))
        return fe

    @staticmethod
    def convert_d2p(dict_data: dict) -> vtkStructuredPoints:
        """Convert field dictionary to structured points"""

        sp = vtkStructuredPoints()
        for v in dict_data.values():
            for name, arr in v.items():
                sp.GetPointData().AddArray(arr)
        return sp

    @staticmethod
    def convert_d2e(field_dim: Tuple[int, int, int], dict_data: dict) -> FieldExtractorCML:
        """Convert field dictionary to a field extractor"""

        fe = FieldExtractorCML()
        fe.setFieldDim(Dim3D(*field_dim))
        sp = FieldStreamer.convert_d2p(dict_data)
        fe.setSimulationData(extract_address_int_from_vtk_object(sp))
        return fe
