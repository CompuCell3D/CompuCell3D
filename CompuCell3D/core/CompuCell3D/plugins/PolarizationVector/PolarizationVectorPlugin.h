

#ifndef POLARIZATIONVECTORPLUGIN_H
#define POLARIZATIONVECTORPLUGIN_H

#include <CompuCell3D/CC3D.h>

#include "PolarizationVector.h"


#include "PolarizationVectorDLLSpecifier.h"

namespace CompuCell3D {

    class CellG;

    class POLARIZATIONVECTOR_EXPORT PolarizationVectorPlugin : public Plugin {

        ExtraMembersGroupAccessor <PolarizationVector> polarizationVectorAccessor;

    public:

        PolarizationVectorPlugin();

        virtual ~PolarizationVectorPlugin();

        ExtraMembersGroupAccessor <PolarizationVector> *
        getPolarizationVectorAccessorPtr() { return &polarizationVectorAccessor; }

        // SimObject interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *simulator);

        void setPolarizationVector(CellG *_cell, float _x, float _y, float _z);

        std::vector<float> getPolarizationVector(CellG *_cell);
    };
};
#endif
