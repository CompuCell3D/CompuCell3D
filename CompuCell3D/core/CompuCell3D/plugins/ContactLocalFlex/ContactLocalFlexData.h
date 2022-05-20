#ifndef CONTACTLOCALFLEXDATA_H
#define CONTACTLOCALFLEXDATA_H

#include "ContactLocalFlexDLLSpecifier.h"
#include <set>
#include <unordered_map>


namespace CompuCell3D {

    class CellG;


    class CONTACTLOCALFLEX_EXPORT ContactLocalFlexData {

    public:
        ContactLocalFlexData() : neighborAddress(0), J(0) {}

        ///have to define < operator if using a class in the set and no < operator is defined for this class
        bool operator<(const ContactLocalFlexData &_rhs) const {
            return neighborAddress < _rhs.neighborAddress;
        }

        CellG *neighborAddress;
        double J;

    };


    class ContactLocalFlexDataContainer {
    public:
        ContactLocalFlexDataContainer() {};

        ~ContactLocalFlexDataContainer() {};
        std::set <ContactLocalFlexData> contactDataContainer; //stores contact energies for cell neighbors
        //add local default values
        std::unordered_map<unsigned char, std::unordered_map<unsigned char, double> > localDefaultContactEnergies;

    };


};
#endif
