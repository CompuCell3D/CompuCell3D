"""
This is a Python prototype of future C++ code.

Permanent implementation will likely reside in CompuCell3D/core/pyinterface/PlayerPythonNew
"""

# todo: add permanent implementation to backend

from enum import Enum
from typing import Dict, List, Optional

from vtkmodules.vtkCommonCorePython import vtkDoubleArray, vtkCharArray, vtkLongArray
from vtkmodules.vtkCommonDataModelPython import vtkStructuredPoints

from cc3d.cpp.CompuCell import cellfield, Point3D, Simulator, floatfield
from cc3d.cpp.PlayerPython import FieldStorage, ScalarFieldCellLevel, VectorFieldCellLevel, Coodrinates3DFloat


class FieldTypeCML(Enum):
    """Enums for field type in field writer"""

    # todo: formalize field type enums somewhere and implement throughout

    CellField = 0
    ConField = 1
    ScalarField = 2
    ScalarFieldCellLevel = 3
    VectorField = 4
    VectorFieldCellLevel = 5


class FieldWriterCML:
    """
    Field writer for simulation-independent data throughput with VTK
    """

    def __init__(self):

        self.sim: Optional[Simulator] = None
        self.fsPtr: Optional[FieldStorage] = None
        self.latticeData: Optional[vtkStructuredPoints] = None
        self.arrayNameVec: List[str] = []
        self.arrayTypeVec: List[FieldTypeCML] = []

    def init(self, sim: Simulator):
        """Initialize the field writer. Must be executed before performing any writing operations"""

        self.sim = sim
        self.latticeData = vtkStructuredPoints()
        fieldDim = self.sim.getPotts().getCellFieldG().getDim()
        self.latticeData.SetDimensions(fieldDim.x, fieldDim.y, fieldDim.z)

    def setFieldStorage(self, _fsPtr: FieldStorage):
        """Set the field storage of the writer"""

        self.fsPtr = _fsPtr

    def getFieldStorage(self) -> FieldStorage:
        """Get the field storage of the writer"""

        return self.fsPtr

    def addCellFieldForOutput(self) -> bool:
        """
        Add the cell fields to the writer for output.

        Adds the cell type field with name "CellType",
        cell id field with name "CellId" and
        cluster id field with name "ClusterId".

        :return: True on success
        :rtype: bool
        """

        cellFieldG: cellfield = self.sim.getPotts().getCellFieldG()
        fieldDim = cellFieldG.getDim()

        typeArray = vtkCharArray()
        typeArray.SetName("CellType")
        self.arrayNameVec.append("CellType")
        self.arrayTypeVec.append(FieldTypeCML.CellField)

        idArray = vtkLongArray()
        idArray.SetName("CellId")
        self.arrayNameVec.append("CellId")
        self.arrayTypeVec.append(FieldTypeCML.CellField)

        clusterIdArray = vtkLongArray()
        clusterIdArray.SetName("ClusterId")
        self.arrayNameVec.append("ClusterId")
        self.arrayTypeVec.append(FieldTypeCML.CellField)

        numberOfValues = fieldDim.x * fieldDim.y * fieldDim.z

        typeArray.SetNumberOfValues(numberOfValues)
        idArray.SetNumberOfValues(numberOfValues)
        clusterIdArray.SetNumberOfValues(numberOfValues)

        offset = 0

        for z in range(fieldDim.z):
            for y in range(fieldDim.y):
                for x in range(fieldDim.x):
                    cell = cellFieldG.get(Point3D(x, y, z))
                    if cell is not None:
                        typeArray.SetValue(offset, chr(cell.type))
                        idArray.SetValue(offset, cell.id)
                        clusterIdArray.SetValue(offset, cell.clusterId)
                    else:
                        typeArray.SetValue(offset, chr(0))
                        idArray.SetValue(offset, 0)
                        clusterIdArray.SetValue(offset, 0)

                    offset += 1

        self.latticeData.GetPointData().AddArray(typeArray)
        self.latticeData.GetPointData().AddArray(idArray)
        self.latticeData.GetPointData().AddArray(clusterIdArray)

        return True

    def addConFieldForOutput(self, _conFieldName: str) -> bool:
        """
        Add a concentration field to the writer for output by name

        :param _conFieldName: name of the field
        :type _conFieldName: str
        :return: True on success
        :rtype: bool
        """

        cellFieldG: cellfield = self.sim.getPotts().getCellFieldG()
        fieldDim = cellFieldG.getDim()

        if _conFieldName in self.sim.getConcentrationFieldNameVector():
            conFieldPtr = self.sim.getConcentrationFieldByName(_conFieldName)
        else:
            return False

        conArray = vtkDoubleArray()
        conArray.SetName(_conFieldName)
        self.arrayNameVec.append(_conFieldName)
        self.arrayTypeVec.append(FieldTypeCML.ConField)

        numberOfValues = fieldDim.x * fieldDim.y * fieldDim.z

        conArray.SetNumberOfValues(numberOfValues)
        offset = 0

        for z in range(fieldDim.z):
            for y in range(fieldDim.y):
                for x in range(fieldDim.x):
                    conArray.SetValue(offset, conFieldPtr.get(Point3D(x, y, z)))
                    offset += 1

        self.latticeData.GetPointData().AddArray(conArray)
        return True

    def addScalarFieldForOutput(self, _scalarFieldName: str) -> bool:
        """
        Add a scalar field to the writer for output by name

        :param _scalarFieldName: name of the field
        :type _scalarFieldName: str
        :return: True on success
        :rtype: bool
        """

        cellFieldG: cellfield = self.sim.getPotts().getCellFieldG()
        fieldDim = cellFieldG.getDim()

        conFieldPtr: floatfield = self.fsPtr.getScalarFieldByName(_scalarFieldName)

        if conFieldPtr is None:
            return False

        conArray = vtkDoubleArray()
        conArray.SetName(_scalarFieldName)
        self.arrayNameVec.append(_scalarFieldName)
        self.arrayTypeVec.append(FieldTypeCML.ScalarField)

        numberOfValues = fieldDim.x * fieldDim.y * fieldDim.z

        conArray.SetNumberOfValues(numberOfValues)
        offset = 0

        for z in range(fieldDim.z):
            for y in range(fieldDim.y):
                for x in range(fieldDim.x):
                    conArray.SetValue(offset, conFieldPtr[x][y][z])
                    offset += 1

        self.latticeData.GetPointData().AddArray(conArray)
        return True

    def addScalarFieldCellLevelForOutput(self, _scalarFieldCellLevelName: str) -> bool:
        """
        Add a cell-level scalar field to the writer for output by name

        :param _scalarFieldCellLevelName: name of the field
        :type _scalarFieldCellLevelName: str
        :return: True on success
        :rtype: bool
        """

        cellFieldG: cellfield = self.sim.getPotts().getCellFieldG()
        fieldDim = cellFieldG.getDim()

        conFieldPtr: ScalarFieldCellLevel = self.fsPtr.getScalarFieldCellLevelFieldByName(_scalarFieldCellLevelName)

        if conFieldPtr is None:
            return False

        conArray = vtkDoubleArray()
        conArray.SetName(_scalarFieldCellLevelName)
        self.arrayNameVec.append(_scalarFieldCellLevelName)
        self.arrayTypeVec.append(FieldTypeCML.ScalarFieldCellLevel)

        numberOfValues = fieldDim.x * fieldDim.y * fieldDim.z

        conArray.SetNumberOfValues(numberOfValues)
        offset = 0

        for z in range(fieldDim.z):
            for y in range(fieldDim.y):
                for x in range(fieldDim.x):
                    cell = cellFieldG.get(Point3D(x, y, z))
                    if cell is not None:
                        mitr = conFieldPtr.find(cell)
                        if mitr != conFieldPtr.end():
                            con = mitr.second
                        else:
                            con = 0.0
                    else:
                        con = 0.0
                    conArray.SetValue(offset, con)
                    offset += 1

        self.latticeData.GetPointData().AddArray(conArray)
        return True

    def addVectorFieldForOutput(self, _vectorFieldName: str) -> bool:
        """
        Add a vector field to the writer for output by name

        :param _vectorFieldName: name of the field
        :type _vectorFieldName: str
        :return: True on success
        :rtype: bool
        """

        cellFieldG: cellfield = self.sim.getPotts().getCellFieldG()
        fieldDim = cellFieldG.getDim()

        vecFieldPtr = self.fsPtr.getVectorFieldFieldByName(_vectorFieldName)

        if vecFieldPtr is None:
            return False

        vecArray = vtkDoubleArray()
        vecArray.SetNumberOfComponents(3)
        vecArray.SetName(_vectorFieldName)
        self.arrayNameVec.append(_vectorFieldName)
        self.arrayTypeVec.append(FieldTypeCML.VectorField)

        numberOfValues = fieldDim.x * fieldDim.y * fieldDim.z

        vecArray.SetNumberOfTuples(numberOfValues)
        offset = 0

        for z in range(fieldDim.z):
            for y in range(fieldDim.y):
                for x in range(fieldDim.x):
                    vx = vecFieldPtr[x][y][z][0]
                    vy = vecFieldPtr[x][y][z][1]
                    vz = vecFieldPtr[x][y][z][2]
                    vecArray.SetTuple3(offset, vx, vy, vz)
                    offset += 1

        self.latticeData.GetPointData().AddArray(vecArray)
        return True

    def addVectorFieldCellLevelForOutput(self, _vectorFieldCellLevelName: str) -> bool:
        """
        Add a cell-level vector field to the writer for output by name

        :param _vectorFieldCellLevelName: name of the field
        :type _vectorFieldCellLevelName: str
        :return: True on success
        :rtype: bool
        """

        cellFieldG: cellfield = self.sim.getPotts().getCellFieldG()
        fieldDim = cellFieldG.getDim()

        vecFieldPtr: VectorFieldCellLevel = self.fsPtr.getVectorFieldCellLevelFieldByName(_vectorFieldCellLevelName)

        if vecFieldPtr is None:
            return False

        vecArray = vtkDoubleArray()
        vecArray.SetNumberOfComponents(3)
        vecArray.SetName(_vectorFieldCellLevelName)
        self.arrayNameVec.append(_vectorFieldCellLevelName)
        self.arrayTypeVec.append(FieldTypeCML.VectorFieldCellLevel)

        numberOfValues = fieldDim.x * fieldDim.y * fieldDim.z

        vecArray.SetNumberOfTuples(numberOfValues)
        offset = 0

        for z in range(fieldDim.z):
            for y in range(fieldDim.y):
                for x in range(fieldDim.x):
                    cell = cellFieldG.get(Point3D(x, y, z))
                    if cell is not None:
                        mitr = vecFieldPtr.find(cell)
                        if mitr != vecFieldPtr.end():
                            vecTmp = mitr.second
                        else:
                            vecTmp = Coodrinates3DFloat(0.0, 0.0, 0.0)
                    else:
                        vecTmp = Coodrinates3DFloat(0.0, 0.0, 0.0)

                    vecArray.SetTuple3(offset, vecTmp.x, vecTmp.y, vecTmp.z)
                    offset += 1

        self.latticeData.GetPointData().AddArray(vecArray)
        return True

    def addFieldForOutput(self, _field_name: str) -> bool:
        """
        Add a field to the writer for output by name. Field type is automatically determined from simulator state

        :param _field_name: name of field
        :type _field_name: str
        :return: True on success
        :rtype: bool
        """

        if _field_name in self.sim.getConcentrationFieldNameVector():
            return self.addConFieldForOutput(_field_name)
        elif _field_name in self.fsPtr.getScalarFieldNameVector():
            return self.addScalarFieldForOutput(_field_name)
        elif _field_name in self.fsPtr.getScalarFieldCellLevelNameVector():
            return self.addScalarFieldCellLevelForOutput(_field_name)
        elif _field_name in self.fsPtr.getVectorFieldNameVector():
            return self.addVectorFieldForOutput(_field_name)
        elif _field_name in self.fsPtr.getVectorFieldCellLevelNameVector():
            return self.addVectorFieldCellLevelForOutput(_field_name)
        return False

    def clear(self) -> None:

        for i in range(len(self.arrayNameVec)):
            self.latticeData.GetPointData().RemoveArray(self.arrayNameVec[i])
        self.arrayNameVec.clear()
