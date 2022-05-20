#ifndef ADHESIONFLEXDATA_H
#define ADHESIONFLEXDATA_H

#include <CompuCell3D/CC3D.h>

#include "AdhesionFlexDLLSpecifier.h"

namespace CompuCell3D {

    class CellG;


    class ADHESIONFLEX_EXPORT AdhesionFlexData {

    public:
        typedef std::vector<float> ContainerType_t;

        AdhesionFlexData() : adhesionMoleculeDensityVec(std::vector<float>(1, 0.0)) {}

        std::vector<float> adhesionMoleculeDensityVec;

        void assignValue(unsigned int _pos, float _value) {
            if (_pos > adhesionMoleculeDensityVec.size() - 1) {
                unsigned int adhesionMoleculeDensityVecSize = adhesionMoleculeDensityVec.size();
                for (unsigned int i = 0; i < _pos - (adhesionMoleculeDensityVecSize - 1); ++i) {
                    adhesionMoleculeDensityVec.push_back(0.);
                }
                adhesionMoleculeDensityVec[_pos] = _value;
            } else {
                adhesionMoleculeDensityVec[_pos] = _value;
            }
        }

        void assignAdhesionMoleculeDensityVector(std::vector<float> &_adhesionVector) {
            adhesionMoleculeDensityVec = _adhesionVector;
        }

        float getValue(unsigned int _pos) {
            if (_pos > (adhesionMoleculeDensityVec.size() - 1)) {
                return 0.;
            } else {
                return adhesionMoleculeDensityVec[_pos];
            }
        }
    };

    class ADHESIONFLEX_EXPORT AdhesionMoleculeDensityData {
    public:
        AdhesionMoleculeDensityData() : density(0.0) {}

        std::string molecule;
        float density;
    };

};
#endif
