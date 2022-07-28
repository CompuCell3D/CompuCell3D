#ifndef NEIGHBORFINDER_H
#define NEIGHBORFINDER_H

#include <vector>

#include "Neighbor.h"

namespace CompuCell3D {

    /**
    * Finds neighbor offsets in 3D space.
    * NeighborFinder uses lazy evaluation and maintains a list of
    * previously accessed neighbor offsets to provide an efficient
    * method for finding neighbors in a discrete 3D grid.
    */
    class NeighborFinder {
        static NeighborFinder *singleton;

        NeighborFinder() : depth(0) {}

        std::vector <Neighbor> neighbors;
        int depth;

    public:

        /**
         * @return A pointer to the single instanceof NeighborFinder.
         */
        static NeighborFinder *getInstance() {
            if (!singleton) singleton = new NeighborFinder();
            return singleton;
        }


        /**
         * Get the ith Neighbor. Neighbor are sorted according to
         * distance.  Neighbors can be accessed as far out as desired.
         * NeighborFinder keeps a list of previously calculated Neighbor(s) for
         * efficiency.  Neighbors are ofsets relative to the origin.  To get
         * a specific neighbor simply add the offset to the original point.
         *
         * @param i The Neighbor index.
         *
         * @return The ith Neighbor.
         */
        Neighbor &getNeighbor(const unsigned int i) const {
            while (i >= neighbors.size())
                ((NeighborFinder *) this)->getMore();
            return const_cast<Neighbor &>(neighbors[i]);
        }

        ~NeighborFinder();

        static void destroy();

    protected:
        /**
         * Find more Neighbors and put them in the list.
         * All the Neighbors at the next level of distance are added to the list
         * at once.
         */
        void getMore();

        /**
         * Takes three coordinates and rotates them around all possible angles
         * accounting for zeros and duplicate numbers.
         *
         * @param nums The coordinates.
         * @param distance The distance at which these coordinates were found.
         */
        void addNeighbors(int nums[3], const double distance);
    };
};
#endif
