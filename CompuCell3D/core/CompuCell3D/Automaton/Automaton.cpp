#include "Automaton.h"

using namespace CompuCell3D;

void Automaton::field3DChange(const Point3D &pt, CellG *newCell,
                              CellG *oldCell) {


    if (newCell) {
        unsigned char currenttype = newCell->type;

        newCell->type = classType->update(pt, newCell);
        if (currenttype != newCell->type) creation(newCell);
        else updateVariables(newCell);
    }
}

