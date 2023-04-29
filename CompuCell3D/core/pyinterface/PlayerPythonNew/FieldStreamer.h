/**
 * @file FieldStreamer.h
 * @author T.J. Sego, Ph.D.
 * @brief Defines features for facilitating moving FieldWriterCML data
 * @date 2022-04-19
 * 
 */
#ifndef FIELDSTREAMER_H
#define FIELDSTREAMER_H

#include "FieldExtractorDLLSpecifier.h"

#include "FieldExtractorCML.h"
#include "FieldWriterCML.h"

#include <vtkAbstractArray.h>

#include <map>

namespace CompuCell3D { 

    struct FIELDEXTRACTOR_EXPORT FieldStreamerData {
        std::vector<std::string> cellFieldNames;
        std::vector<std::string> concFieldNames;
        std::vector<std::string> scalarFieldNames;
        std::vector<std::string> scalarFieldCellLevelNames;
        std::vector<std::string> vectorFieldNames;
        std::vector<std::string> vectorFieldCellLevelNames;
        std::vector<std::string> linksNames;
        std::vector<std::string> linksInternalNames;
        std::vector<std::string> anchorsNames;
        Dim3D fieldDim;
        std::string data;

        std::vector<std::string> getFieldNames();
    };

    class FIELDEXTRACTOR_EXPORT FieldStreamer {

        FieldStreamerData data;
        vtkStructuredPoints *points;

    public:

        FieldStreamer();
        FieldStreamer(const FieldStreamerData &_data);
        ~FieldStreamer();

        void loadData(const FieldStreamerData &_data);

        static FieldStreamerData dump(FieldWriterCML *fieldWriter);

        vtk_obj_addr_int_t getPointsAddr() { return (vtk_obj_addr_int_t)points; };
        Dim3D getFieldDim() { return data.fieldDim; };

    };

};

#endif // FIELDSTREAMER_H