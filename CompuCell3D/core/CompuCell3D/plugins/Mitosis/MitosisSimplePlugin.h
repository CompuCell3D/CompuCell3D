#ifndef MITOSISSIMPLEPLUGIN_H
#define MITOSISSIMPLEPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "MitosisPlugin.h"
#include "MitosisDLLSpecifier.h"


namespace CompuCell3D {
    class Potts3D;

    class PixelTracker;

    class PixelTrackerPlugin;

    class MITOSIS_EXPORT OrientationVectorsMitosis {
    public:
        OrientationVectorsMitosis() {}


        Coordinates3D<double> semiminorVec;
        Coordinates3D<double> semimajorVec;


    };

    class MITOSIS_EXPORT MitosisSimplePlugin : public MitosisPlugin {
    public:

        typedef bool (MitosisSimplePlugin::*doDirectionalMitosis2DPtr_t)();

        doDirectionalMitosis2DPtr_t doDirectionalMitosis2DPtr;

        typedef OrientationVectorsMitosis (MitosisSimplePlugin::*getOrientationVectorsMitosis2DPtr_t)(CellG *);

        getOrientationVectorsMitosis2DPtr_t getOrientationVectorsMitosis2DPtr;

        MitosisSimplePlugin();

        virtual ~MitosisSimplePlugin();

        bool divideAlongMinorAxisFlag;
        bool divideAlongMajorAxisFlag;
        bool flag3D;


        ExtraMembersGroupAccessor <PixelTracker> *pixelTrackerAccessorPtr;
        PixelTrackerPlugin *pixelTrackerPlugin;

        virtual void handleEvent(CC3DEvent &_event);

        // SimObject interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        void setDivideAlongMinorAxis();

        void setDivideAlongMajorAxis();


        OrientationVectorsMitosis getOrientationVectorsMitosis(CellG *);

        OrientationVectorsMitosis getOrientationVectorsMitosis2D_xy(CellG *);

        OrientationVectorsMitosis getOrientationVectorsMitosis2D_xz(CellG *);

        OrientationVectorsMitosis getOrientationVectorsMitosis2D_yz(CellG *);

        OrientationVectorsMitosis getOrientationVectorsMitosis3D(CellG *);

        bool doDirectionalMitosis();

        bool doDirectionalMitosis2D_xy();

        bool doDirectionalMitosis2D_xz();

        bool doDirectionalMitosis2D_yz();

        bool doDirectionalMitosis3D();

        bool doDirectionalMitosisOrientationVectorBased(double _nx, double _ny, double _nz);

        void setMitosisFlag(bool _flag);

        bool getMitosisFlag();

    };
};
#endif
