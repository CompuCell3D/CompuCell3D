
#ifndef CONNECTIVITYGLOBALPLUGIN_H
#define CONNECTIVITYGLOBALPLUGIN_H

#include <CompuCell3D/CC3D.h>

#include "ConnectivityGlobalData.h"

#include "ConnectivityGlobalDLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {
    class Potts3D;

    class Automaton;

    class BoundaryStrategy;


    class CONNECTIVITYGLOBAL_EXPORT ConnectivityGlobalPlugin : public Plugin, public EnergyFunction {

        //Energy Function data
    private:

        ExtraMembersGroupAccessor <ConnectivityGlobalData> connectivityGlobalDataAccessor;

        unsigned int maxNeighborIndex;
        unsigned int max_neighbor_index_local_search;
        BoundaryStrategy *boundaryStrategy;

        Potts3D *potts;
        std::unordered_map<unsigned char, double> penaltyMap;
        unsigned char maxTypeId;
        bool doNotPrecheckConnectivity;
        bool fast_algorithm;

        typedef double (ConnectivityGlobalPlugin::*changeEnergyFcnPtr_t)(const Point3D &pt, const CellG *newCell,
                                                                         const CellG *oldCell);

        changeEnergyFcnPtr_t changeEnergyFcnPtr;


    public:
        ConnectivityGlobalPlugin();

        virtual ~ConnectivityGlobalPlugin();

        //Plugin interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        ExtraMembersGroupAccessor <ConnectivityGlobalData> *
        getConnectivityGlobalDataPtr() { return &connectivityGlobalDataAccessor; }

        void setConnectivityStrength(CellG *_cell, double _connectivityStrength);

        double getConnectivityStrength(CellG *_cell);


        virtual std::string toString();

        //EnergyFunction interface

        virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        double changeEnergyLegacy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        double changeEnergyFast(const Point3D &pt, const CellG *newCell, const CellG *oldCell);


        bool checkIfCellIsFragmented(const CellG *cell, Point3D cellPixel);

        bool
        check_local_connectivity(const Point3D &pt, const CellG *cell, unsigned int max_neighbor_index_local_search,
                                 bool add_pt_to_bfs);

        //SteerableObject interface

        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();


    };
};
#endif
