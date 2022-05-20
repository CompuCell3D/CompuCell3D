#ifndef MITOSISPLUGIN_H
#define MITOSISPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "MitosisDLLSpecifier.h"

namespace CompuCell3D {
    class Potts3D;

    class ParallelUtilsOpenMP;

    class BoundaryStrategy;

    class MITOSIS_EXPORT MitosisPlugin : public Plugin, public CellGChangeWatcher,
                                         public Stepper {
    protected:

        Potts3D *potts;
        ParallelUtilsOpenMP *pUtils;
        unsigned int doublingVolume;

        //CellG *childCell;
        //CellG *parentCell;

        std::vector<CellG *> childCellVec;
        std::vector<CellG *> parentCellVec;



        //Point3D splitPt;
        //bool split;
        //bool on;

        std::vector <Point3D> splitPtVec;
        std::vector<short> splitVec; //using shorts instead of bool because vector<bool> has special implementation  not suitable for this plugin
        std::vector<short> onVec;
        std::vector<short> mitosisFlagVec;


        unsigned int maxNeighborIndex;
        BoundaryStrategy *boundaryStrategy;

    public:

        MitosisPlugin();

        virtual ~MitosisPlugin();

        // SimObject interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual void handleEvent(CC3DEvent &_event);

        // CellChangeWatcher interface
        virtual void field3DChange(const Point3D &pt, CellG *newCell,
                                   CellG *oldCell);

        // Stepper interface
        virtual void step();

        //Steerable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();

        // Functions to turn on and off
        virtual void turnOn();

        virtual void turnOff();
        //virtual void turnOnAll();
        //virtual void turnOffAll();

        virtual bool doMitosis();///actually does mitosis - returns true if mitosis was done , false otherwise
        virtual void updateAttributes();///updates some of the cell attributes
        CellG *getChildCell();

        CellG *getParentCell();

        void setPotts(Potts3D *_potts) { potts = _potts; }

        unsigned int getDoublingVolume() { return doublingVolume; }

        void setDoublingVolume(unsigned int _doublingVolume) { doublingVolume = _doublingVolume; }


    };
};
#endif
