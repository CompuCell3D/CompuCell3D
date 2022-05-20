
#ifndef ORIENTEDCONTACTPLUGIN_H
#define ORIENTEDCONTACTPLUGIN_H

#include <CompuCell3D/CC3D.h>

#include "OrientedContactDLLSpecifier.h"

class CC3DXMLElement;
namespace CompuCell3D {


    class Potts3D;

    class Automaton;

    class BoundaryStrategy;

    class Simulator;

    class ORIENTEDCONTACT_EXPORT OrientedContactPlugin : public Plugin, public EnergyFunction {
        //Energy Function data
        CC3DXMLElement *xmlData;
        Potts3D *potts;
        Simulator *sim;

        typedef std::unordered_map<unsigned char, std::unordered_map<unsigned char, double> > orientedContactEnergyArray_t;

        orientedContactEnergyArray_t orientedContactEnergyArray;

        std::string autoName;
        double depth;
        double alpha;

        Automaton *automaton;
        bool weightDistance;
        unsigned int maxNeighborIndex;
        BoundaryStrategy *boundaryStrategy;


    public:
        OrientedContactPlugin();

        virtual ~OrientedContactPlugin();

        //EnergyFunction interface
        virtual double changeEnergy(const Point3D &pt, const CellG *newCell,
                                    const CellG *oldCell);

        //Plugin interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *simulator);

        virtual std::string toString();

        //Steerable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();
        //Energy Function Methods

        double getOrientation(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        double getMediumOrientation(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        /**
        * @return The orientedContact energy between cell1 and cell2.
        */
        double orientedContactEnergy(const CellG *cell1, const CellG *cell2);

        /**
        * Sets the orientedContact energy for two cell types.  A -1 type is interpreted
        * as the medium.
        */
        void setOrientedContactEnergy(const std::string typeName1,
                                      const std::string typeName2, const double energy);

    };
};
#endif
