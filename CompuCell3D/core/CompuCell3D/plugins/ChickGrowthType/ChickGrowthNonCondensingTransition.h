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

#ifndef CHICKGROWTHNONCONDENSINGTRANSITION_H
#define CHICKGROWTHNONCONDENSINGTRANSITION_H

#include <CompuCell3D/Automaton/Transition.h>
#include <CompuCell3D/plugins/ChickGrowthType/ChickGrowthTypePlugin.h>
#include <CompuCell3D/Simulator.h>

namespace CompuCell3D {
  
  class Cell;

  /** 
   * An interface for transitions between cell types. In computer talk this
   * would be cell state rather than type.
   */
  class ChickGrowthNonCondensingTransition : public Transition {
    
  public:
   
    ChickGrowthNonCondensingTransition(char cellType) : Transition(cellType) {} 
    /**
     * @param pt The point in the grid. 
     * @param cell The cell to query.
     * 
     * @return True of the condition is true false otherwise
     */
    bool checkCondition(const Point3D& pt, CellG *cell)
    {
        ChickGrowthTypePlugin* chickGrowthTypePlugin = (ChickGrowthTypePlugin*)Simulator::pluginManager.get("ChickGrowthType");
        float concentration = chickGrowthTypePlugin->getConcentration(pt);
        float threshold = chickGrowthTypePlugin->getThreshold();
        return ( (const_cast<CellG*>(cell)->type == 2) && (concentration < threshold) );
    };
  };
};
#endif
