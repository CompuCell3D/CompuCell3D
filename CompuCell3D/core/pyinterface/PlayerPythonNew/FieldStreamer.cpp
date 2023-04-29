/**
 * @file FieldStreamer.cpp
 * @author T.J. Sego, Ph.D.
 * @brief Defines features for facilitating moving FieldWriterCML data
 * @date 2022-04-19
 * 
 */

#include "FieldStreamer.h"

#include <vtkArray.h>
#include <vtkArrayReader.h>
#include <vtkArrayWriter.h>
#include <vtkCharArray.h>
#include <vtkDoubleArray.h>
#include <vtkLongArray.h>
#include <vtkPointData.h>
#include <vtkStructuredPoints.h>
#include <vtkStructuredPointsReader.h>
#include <vtkStructuredPointsWriter.h>

using namespace CompuCell3D;


///////////////////////
// FieldStreamerData //
///////////////////////


std::vector<std::string> FieldStreamerData::getFieldNames() {
    std::vector<std::string> result;
    for(auto &s : cellFieldNames) 
        result.push_back(s);
    for(auto &s : concFieldNames) 
        result.push_back(s);
    for(auto &s : scalarFieldNames) 
        result.push_back(s);
    for(auto &s : scalarFieldCellLevelNames) 
        result.push_back(s);
    for(auto &s : vectorFieldNames) 
        result.push_back(s);
    for(auto &s : vectorFieldCellLevelNames) 
        result.push_back(s);
    for(auto &s : linksNames) 
        result.push_back(s);
    for(auto &s : linksInternalNames) 
        result.push_back(s);
    for(auto &s : anchorsNames) 
        result.push_back(s);
    return result;
}


///////////////////
// FieldStreamer //
///////////////////


FieldStreamer::FieldStreamer() : 
    points(0) 
{}

FieldStreamer::FieldStreamer(const FieldStreamerData &_data) {
    loadData(_data);
}

FieldStreamer::~FieldStreamer() {
    if(points) 
        points = 0;
}

void FieldStreamer::loadData(const FieldStreamerData &_data) {
    data = FieldStreamerData();
    data.cellFieldNames = _data.cellFieldNames;
    data.concFieldNames = _data.concFieldNames;
    data.scalarFieldNames = _data.scalarFieldNames;
    data.scalarFieldCellLevelNames = _data.scalarFieldCellLevelNames;
    data.vectorFieldNames = _data.vectorFieldNames;
    data.vectorFieldCellLevelNames = _data.vectorFieldCellLevelNames;
    data.linksNames = _data.linksNames;
    data.linksInternalNames = _data.linksInternalNames;
    data.anchorsNames = _data.anchorsNames;
    data.fieldDim = _data.fieldDim;

    vtkStructuredPointsReader *reader = vtkStructuredPointsReader::New();
    reader->ReadFromInputStringOn();
    reader->SetInputString(_data.data);
    reader->Update();
    points = reader->GetOutput();
}

FieldStreamerData FieldStreamer::dump(FieldWriterCML *fieldWriter) {
    FieldStreamerData fsd;

    if(!fieldWriter) 
        return fsd;

    fsd.fieldDim = fieldWriter->getFieldDim();

    for(unsigned int i = 0; i < fieldWriter->numFields(); i++) {
        std::string fieldName = fieldWriter->getFieldName(i);
        FieldTypeCML fieldType = fieldWriter->getFieldType(i);

        switch (fieldType)
        {
        case FieldTypeCML::FieldTypeCML_CellField:
            fsd.cellFieldNames.push_back(fieldName);
            break;
        case FieldTypeCML::FieldTypeCML_ConField:
            fsd.concFieldNames.push_back(fieldName);
            break;
        case FieldTypeCML::FieldTypeCML_ScalarField:
            fsd.scalarFieldNames.push_back(fieldName);
            break;
        case FieldTypeCML::FieldTypeCML_ScalarFieldCellLevel:
            fsd.scalarFieldCellLevelNames.push_back(fieldName);
            break;
        case FieldTypeCML::FieldTypeCML_VectorField:
            fsd.vectorFieldNames.push_back(fieldName);
            break;
        case FieldTypeCML::FieldTypeCML_VectorFieldCellLevel:
            fsd.vectorFieldCellLevelNames.push_back(fieldName);
            break;
        case FieldTypeCML::FieldTypeCML_Links:
            fsd.linksNames.push_back(fieldName);
            break;
        case FieldTypeCML::FieldTypeCML_LinksInternal:
            fsd.linksInternalNames.push_back(fieldName);
            break;
        case FieldTypeCML::FieldTypeCML_Anchors:
            fsd.anchorsNames.push_back(fieldName);
            break;
        default:
            break;
        }
    }

    vtkStructuredPointsWriter *writer = vtkStructuredPointsWriter::New();
    writer->SetInputData(fieldWriter->latticeData);
    writer->WriteToOutputStringOn();
    writer->Update();
    fsd.data = writer->GetOutputStdString();

    return fsd;
}
