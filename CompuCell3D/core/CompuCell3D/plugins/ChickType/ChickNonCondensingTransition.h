

#ifndef CHICKNONCONDENSINGTRANSITION_H
#define CHICKNONCONDENSINGTRANSITION_H

#include <CompuCell3D/Automaton/Transition.h>
#include <CompuCell3D/plugins/ChickType/ChickTypePlugin.h>
#include <CompuCell3D/Simulator.h>

namespace CompuCell3D {
  
  class Cell;

  /** 
   * An interface for transitions between cell types. In computer talk this
   * would be cell state rather than type.
   */
  class ChickNonCondensingTransition : public Transition {
    
  public:
   
    ChickNonCondensingTransition(char cellType) : Transition(cellType) {} 
    /**
     * @param pt The point in the grid. 
     * @param cell The cell to query.
     * 
     * @return True of the condition is true false otherwise
     */
    bool checkCondition(const Point3D& pt, CellG *cell)
    {
        ChickTypePlugin* chickTypePlugin = (ChickTypePlugin*)Simulator::pluginManager.get("ChickType");
        float concentration = chickTypePlugin->getConcentration(pt);
        float threshold = chickTypePlugin->getThreshold();
        return ( (const_cast<CellG*>(cell)->type == 2) && (concentration < threshold) );
    };
  };
};
#endif
