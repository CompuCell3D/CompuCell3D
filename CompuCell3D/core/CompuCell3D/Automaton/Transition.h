#ifndef TRANSITION_H
#define TRANSITION_H


namespace CompuCell3D {

    class Point3D;

    class CellG;

    /**
     * An interface for transitions between cell types. In computer talk this
     * would be cell state rather than type.
     */
    class Transition {
        unsigned char cellType;

    public:
        /**
         * @param cellType The cell type to which the transition goes.
         */
        Transition(const unsigned char cellType) : cellType(cellType) {}

        unsigned char getCellType() { return cellType; }

        /**
         * @param cell The cell to query.
         *
         * @return True of the condition is true false otherwise
         */
        virtual bool checkCondition(const Point3D &pt, CellG *cell) = 0;
    };
};
#endif
