#ifndef SECRETIONPLUGIN_H
#define SECRETIONPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "SecretionDataP.h"
#include "SecretionDLLSpecifier.h"

class CC3DXMLElement;
namespace CompuCell3D {

    class Potts3D;

    class CellG;

    class Steppable;

    class Simulator;

    class Automaton;

    class BoundaryStrategy;

    class BoxWatcher;

    class BoundaryPixelTrackerPlugin;

    class PixelTrackerPlugin;

    class FieldSecretor;

    class ParallelUtilsOpenMP;

    template<typename Y>
    class WatchableField3D;

    template<typename Y>
    class Field3DImpl;


    class SECRETION_EXPORT SecretionPlugin : public Plugin, public FixedStepper {
        Potts3D *potts;
        Simulator *sim;
        CC3DXMLElement *xmlData;

        std::vector <SecretionDataP> secretionDataPVec;
        Dim3D fieldDim;
        WatchableField3D<CellG *> *cellFieldG;
        Automaton *automaton;
        BoxWatcher *boxWatcherSteppable;
        BoundaryPixelTrackerPlugin *boundaryPixelTrackerPlugin;
        PixelTrackerPlugin *pixelTrackerPlugin;

        ParallelUtilsOpenMP *pUtils;
        BoundaryStrategy *boundaryStrategy;
        unsigned int maxNeighborIndex;
        bool disablePixelTracker;
        bool disableBoundaryPixelTracker;

    public:
        SecretionPlugin();

        virtual ~SecretionPlugin();

        typedef void (SecretionPlugin::*secrSingleFieldFcnPtr_t)(unsigned int idx);

        ///SimObject interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *simulator);

        Field3D<float> *getConcentrationFieldByName(std::string _fieldName);

        void secreteSingleField(unsigned int idx);

        void secreteOnContactSingleField(unsigned int idx);

        void secreteConstantConcentrationSingleField(unsigned int idx);

        FieldSecretor getFieldSecretor(std::string _fieldName);

        // Stepper interface
        virtual void step();

        //SteerableObject interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();

    };
};
#endif

