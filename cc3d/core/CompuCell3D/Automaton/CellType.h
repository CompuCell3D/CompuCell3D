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

#ifndef CELLTYPE_H
#define CELLTYPE_H



#include <BasicUtils/BasicArray.h>

namespace CompuCell3D {

  class Transition;
  class Point3D;
  class CellG;
  
  /** 
   * Represents a cell type.  In computer science terms we would call this
   * cell state rather than type.
   */
  class CellType {
    BasicArray<Transition *> transitions;
    
    
  public:
    CellType()  {}

    /** 
     * Add a transition to another CellType
     * 
     * @param transition 
     * 
     * @return The index or id of this transition.
     */
    unsigned char addTransition(Transition *transition) {
      return (unsigned char)transitions.put(transition);
    }

    /** 
     * This function will throw a BasicException if id is out of range.
     *
     * @param id The Transition index or id.
     * 
     * @return A pointer to the transition.
     */
    Transition *getTransition(unsigned char id) {return transitions[id];}

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
