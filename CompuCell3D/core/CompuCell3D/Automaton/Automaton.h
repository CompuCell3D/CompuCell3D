#ifndef AUTOMATON_H
#define AUTOMATON_H

#include <CompuCell3D/Potts3D/Potts3D.h>
#include "CellType.h"

#include <CompuCell3D/Potts3D/CellGChangeWatcher.h>
#include <CompuCell3D/plugins/CellType/CellTypeG.h>
#include <iostream>

#include <vector>
#include <string>


namespace CompuCell3D {
    class Potts3D;

    class Cell;

    class Automaton : public CellGChangeWatcher {

    protected:
        Potts3D *potts;

        CellType *classType;

    public:
        Automaton() {}

        virtual ~Automaton() { if (classType) delete classType; };


        virtual void creation(CellG *newCell) {}

        virtual void updateVariables(CellG *newCell) {}

        virtual void field3DChange(const Point3D &pt, CellG *newCell,
                                   CellG *oldCell);


        virtual unsigned char getCellType(const CellG *) const = 0;

        virtual std::string getTypeName(const char type) const = 0;

        virtual unsigned char getTypeId(const std::string typeName) const = 0;

        virtual unsigned char getMaxTypeId() const = 0;

        virtual const std::vector<unsigned char> getTypeIds() const = 0;
    };
};
#endif
