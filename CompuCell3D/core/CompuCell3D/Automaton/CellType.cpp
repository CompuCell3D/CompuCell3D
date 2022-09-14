

#include "CellType.h"

#include "Transition.h"

#include <CompuCell3D/Field3D/Point3D.h>
#include <CompuCell3D/Potts3D/Cell.h>

#include <iostream>

using namespace std;

using namespace CompuCell3D;

unsigned char CellType::update(const Point3D &pt, CellG *cell) {
    if (cell) {
        for (unsigned int i = 0; i < transitions.size(); i++) {
            if (transitions[i]->checkCondition(pt, cell)) {
                cell->type = transitions[i]->getCellType();
                break;
            }
        }
        return cell->type;
    } else
        return 0;
}

