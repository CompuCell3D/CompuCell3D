

#ifndef CELLTYPE_H
#define CELLTYPE_H

#include <vector>

#include <CompuCell3D/CC3DExceptions.h>


namespace CompuCell3D {

    class Transition;

    class Point3D;

    class CellG;

    /**
     * Represents a cell type.  In computer science terms we would call this
     * cell state rather than type.
     */
    class CellType {
        std::vector<Transition *> transitions;


    public:
        CellType() {}

        /**
         * Add a transition to another CellType
         *
         * @param transition
         *
         * @return The index or id of this transition.
         */
        unsigned char addTransition(Transition *transition) {
            transitions.push_back(transition);
            return (unsigned char) (transitions.size() - 1);
        }

        /**
         * This function will throw a CC3DException if id is out of range.
         *
         * @param id The Transition index or id.
         *
         * @return A pointer to the transition.
         */
        Transition *getTransition(unsigned char id) {
            if (id >= transitions.size()) throw CC3DException(std::string("Index out of range."));
            return transitions[id];
        }

        /**
         * Update cells state.
         *
         * @param cell The cell to be updated.
         *
         * @return The new CellType id.
         */
        unsigned char update(const Point3D &pt, CellG *cell);
    };
};
#endif
