#ifndef CONNECTIVITYPLUGIN_H
#define CONNECTIVITYPLUGIN_H

#include <CompuCell3D/CC3D.h>
#include "ConnectivityDLLSpecifier.h"


class CC3DXMLElement;

namespace CompuCell3D {
    class Potts3D;

    class Automaton;

    class BoundaryStrategy;


    class CONNECTIVITY_EXPORT ConnectivityPlugin : public Plugin, public EnergyFunction {

        //Energy Function data
    private:
        // bool mediumFlag;
        unsigned int numberOfNeighbors;
        std::vector<int> offsetsIndex; //this vector will contain indexes of offsets in the neighborListVector
        // so that accessing
        //them using offsetsindex will ensure correct clockwise ordering
        //std::vector<Point3D> n;
        unsigned int maxNeighborIndex;
        BoundaryStrategy *boundaryStrategy;

        Potts3D *potts;
        double penalty;


    public:
        ConnectivityPlugin();

        virtual ~ConnectivityPlugin();

        //Plugin interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual std::string toString();

        //EnergyFunction interface
        virtual double changeEnergy(const Point3D &pt, const CellG *newCell,
                                    const CellG *oldCell);


        //SteerableObject interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        //Energy function methods
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
