/*************************************************************************
 *    CompuCell - A software framework for multimodel simulations of     *
 * biocomplexity problems Copyright (C) 2003 University of Notre Dame,   *
 *                             Indiana                                   *
 *                                                                       *
 * This program is free software; IF YOU AGREE TO CITE USE OF CompuCell  *
 *  IN ALL RELATED RESEARCH PUBLICATIONS according to the terms of the   *
 *  CompuCell GNU General Public License RIDER you can redistribute it   *
 * and/or modify it under the terms of the GNU General Public License as *
 *  published by the Free Software Foundation; either version 2 of the   *
 *         License, or (at your option) any later version.               *
 *                                                                       *
 * This program is distributed in the hope that it will be useful, but   *
 *      WITHOUT ANY WARRANTY; without even the implied warranty of       *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU    *
 *             General Public License for more details.                  *
 *                                                                       *
 *  You should have received a copy of the GNU General Public License    *
 *     along with this program; if not, write to the Free Software       *
 *      Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.        *
 *************************************************************************/

#ifndef NEIGHBORFINDER_H
#define NEIGHBORFINDER_H

#include <BasicUtils/BasicArray.h>

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
		BasicArray<Neighbor> neighbors;
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
				while (i >= neighbors.getSize()) 
					((NeighborFinder *)this)->getMore();
				return const_cast<Neighbor&>(neighbors[i]);
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
