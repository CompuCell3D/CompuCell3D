

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
