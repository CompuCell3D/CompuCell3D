
#ifndef CONNECTIVITYLOCALFLEXPLUGIN_H
#define CONNECTIVITYLOCALFLEXPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "ConnectivityLocalFlexData.h"

#include "ConnectivityLocalFlexDLLSpecifier.h"


namespace CompuCell3D {

    class CellG;

    class Potts3D;

    class Automaton;

    class BoundaryStrategy;

    class Simulator;

    class CONNECTIVITYLOCALFLEX_EXPORT  ConnectivityLocalFlexPlugin : public Plugin, public EnergyFunction {

        ExtraMembersGroupAccessor <ConnectivityLocalFlexData> connectivityLocalFlexDataAccessor;
        //Energy Function data
        Potts3D *potts;

        unsigned int numberOfNeighbors;
        std::vector<int> offsetsIndex; //this vector will contain indexes of offsets in the neighborListVector
        // so that accessing
        //them using offsets index will ensure correct clockwise ordering
        //std::vector<Point3D> n;
        unsigned int maxNeighborIndex;
        BoundaryStrategy *boundaryStrategy;

    public:
        ConnectivityLocalFlexPlugin();

        virtual ~ConnectivityLocalFlexPlugin();

        //EnergyFunction interface
        virtual double changeEnergy(const Point3D &pt, const CellG *newCell,
                                    const CellG *oldCell);

        //Plugin interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData);

        ExtraMembersGroupAccessor <ConnectivityLocalFlexData> *
        getConnectivityLocalFlexDataPtr() { return &connectivityLocalFlexDataAccessor; }

        void setConnectivityStrength(CellG *_cell, double _connectivityStrength);

        double getConnectivityStrength(CellG *_cell);

        //Steerable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        virtual std::string toString();

        //EnergyFunction methods
    protected:
        /**
         * @return The index used for ordering connectivity energies in the map.
         */
        void addUnique(CellG *, std::vector<CellG *> &);

        void initializeNeighborsOffsets();

        void orderNeighborsClockwise(Point3D &_midPoint, const std::vector <Point3D> &_offsets,
                                     std::vector<int> &_offsetsIndex);


    };
};
#endif
