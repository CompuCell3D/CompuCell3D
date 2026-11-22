#ifndef COMPUCELL3D_CONCENTRATION_FIELD_IO_H
#define COMPUCELL3D_CONCENTRATION_FIELD_IO_H

#include <CompuCell3D/Field3D/Dim3D.h>

#include <fstream>
#include <sstream>
#include <string>

#include <CompuCell3D/CC3DExceptions.h>
#include <Logger/CC3DLogger.h>

namespace CompuCell3D {

inline bool isAbsoluteConcentrationFieldPath(const std::string &path) {
    return !path.empty() &&
           (path[0] == '/' || path[0] == '\\' ||
            (path.size() > 1 && path[1] == ':'));
}

inline std::string resolveConcentrationFieldPath(const std::string &fileName,
                                                  const std::string &basePath) {
    if (basePath.empty() || isAbsoluteConcentrationFieldPath(fileName)) {
        return fileName;
    }

    return basePath + "/" + fileName;
}

template<class ConcentrationField>
void read_field(const std::string &fileName,
                const std::string &basePath,
                const Dim3D &fieldDim,
                ConcentrationField *concentrationField,
                bool zeroField = true) {
    const std::string resolvedPath = resolveConcentrationFieldPath(fileName, basePath);
    std::ifstream in(resolvedPath.c_str());

    if (!in.is_open()) {
        std::ostringstream message;
        message << "Chemical concentration field file cannot be found or opened. "
                << "Requested path: '" << fileName << "'. "
                << "Resolved path: '" << resolvedPath << "'.";
        CC3D_Log(LOG_ERROR) << message.str();
        throw CC3DException(message.str());
    }

    Point3D pt;
    float concentration;

    if (zeroField) {
        for (pt.z = 0; pt.z < fieldDim.z; ++pt.z)
            for (pt.y = 0; pt.y < fieldDim.y; ++pt.y)
                for (pt.x = 0; pt.x < fieldDim.x; ++pt.x)
                    concentrationField->set(pt, 0);
    }

    while (!in.eof()) {
        in >> pt.x >> pt.y >> pt.z >> concentration;
        if (!in.fail())
            concentrationField->set(pt, concentration);
    }
}

} // namespace CompuCell3D

#endif
