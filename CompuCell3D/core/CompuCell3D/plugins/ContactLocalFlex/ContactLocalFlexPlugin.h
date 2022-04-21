

#ifndef CONTACTLOCALFLEXPLUGIN_H
#define CONTACTLOCALFLEXPLUGIN_H


#include <CompuCell3D/CC3D.h>

#include "ContactLocalFlexData.h"

#include "ContactLocalFlexDLLSpecifier.h"

class CC3DXMLElement;

namespace CompuCell3D {

    class Simulator;

    class Potts3D;

    class Automaton;

    class ContactLocalFlexDataContainer;

    class BoundaryStrategy;


    class CONTACTLOCALFLEX_EXPORT ContactLocalFlexPlugin
            : public Plugin, public CellGChangeWatcher, public EnergyFunction {


        ParallelUtilsOpenMP *pUtils;
        ParallelUtilsOpenMP::OpenMPLock_t *lockPtr;

        ExtraMembersGroupAccessor <ContactLocalFlexDataContainer> contactDataContainerAccessor;
        Potts3D *potts;
        Simulator *sim;

        void updateContactEnergyData(CellG *_cell);

        bool initializadContactData;


        // EnergyFunction Data

        typedef std::unordered_map<unsigned char, std::unordered_map<unsigned char, double> > contactEnergyArray_t;

        contactEnergyArray_t contactEnergyArray;

        std::string autoName;
        double depth;

        Automaton *automaton;
        bool weightDistance;

        unsigned int maxNeighborIndex;
        BoundaryStrategy *boundaryStrategy;
        CC3DXMLElement *xmlData;

    public:
        ContactLocalFlexPlugin();

        virtual ~ContactLocalFlexPlugin();

        ExtraMembersGroupAccessor <ContactLocalFlexDataContainer> *
        getContactDataContainerAccessorPtr() { return &contactDataContainerAccessor; }

        void initializeContactLocalFlexData();

        //CellGCellwatcher interface
        virtual void field3DChange(const Point3D &pt, CellG *newCell, CellG *oldCell);

        //EnergyFunction interface
        virtual double changeEnergy(const Point3D &pt, const CellG *newCell, const CellG *oldCell);

        //Plugin interface
        virtual void init(Simulator *simulator, CC3DXMLElement *_xmlData = 0);

        virtual void extraInit(Simulator *simulator);

        virtual std::string toString();

        //Steerable interface
        virtual void update(CC3DXMLElement *_xmlData, bool _fullInitFlag = false);

        virtual std::string steerableName();

        /**
         * @return The contact energy between cell1 and cell2.
         */
        double contactEnergy(const CellG *cell1, const CellG *cell2);

        double defaultContactEnergy(const CellG *cell1, const CellG *cell2);

        /**
         * Sets the contact energy for two cell types.  A -1 type is interpreted
         * as the medium.
         */
        void setContactEnergy(const std::string typeName1,
                              const std::string typeName2, const double energy);

    protected:
        /**
         * @return The index used for ordering contact energies in the map.
         */
        int getIndex(const int type1, const int type2) const;
    };


};

#endif
