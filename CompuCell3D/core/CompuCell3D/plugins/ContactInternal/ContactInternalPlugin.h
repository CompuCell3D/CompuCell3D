

#ifndef CONTACTINTERNALPLUGIN_H
#define CONTACTINTERNALPLUGIN_H

#include <CompuCell3D/CC3D.h>

#include "ContactInternalDLLSpecifier.h"

class CC3DXMLElement;
namespace CompuCell3D {

    class Potts3D;

    class Automaton;

    class BoundaryStrategy;


    class CONTACTINTERNAL_EXPORT ContactInternalPlugin : public Plugin, public EnergyFunction {
        //Energy function data
        Potts3D *potts;
        CC3DXMLElement *xmlData;
        typedef std::map<int, double> contactEnergies_t;
        typedef std::unordered_map<unsigned char, std::unordered_map<unsigned char, double> > contactEnergyArray_t;


        contactEnergyArray_t internalEnergyArray;

        std::string autoName;
        double depth;

        Automaton *automaton;
        bool weightDistance;

        unsigned int maxNeighborIndex;
        BoundaryStrategy *boundaryStrategy;


    public:
        ContactInternalPlugin();

        virtual ~ContactInternalPlugin();

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

        //Energy Function methods

        /**
         * @return The contact energy between cell1 and cell2.
         */

        double internalEnergy(const CellG *cell1, const CellG *cell2);

        /**
         * Sets the contact energy for two cell types.  A -1 type is interpreted
         * as the medium.
         */

        void setContactInternalEnergy(const std::string typeName1,
                                      const std::string typeName2, const double energy);

    };
};
#endif
