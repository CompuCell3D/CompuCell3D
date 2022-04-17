

#ifndef CHICKGROWTHCONDENSINGTRANSITION_H
#define CHICKGROWTHCONDENSINGTRANSITION_H

#include <CompuCell3D/Automaton/Transition.h>
#include <CompuCell3D/plugins/ChickGrowthType/ChickGrowthTypePlugin.h>
#include <CompuCell3D/plugins/Growth/GrowthPlugin.h>
#include <CompuCell3D/Simulator.h>

namespace CompuCell3D {
  
  class Cell;

  /** 
   * An interface for transitions between cell types. In computer talk this
   * would be cell state rather than type.
   */
  class ChickGrowthCondensingTransition : public Transition {
    
  public:
    
     ChickGrowthCondensingTransition(char cellType) : Transition(cellType) {}
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
        bool active = true;
        
        // If the growth plugin is included, the domain is growing
        GrowthPlugin* growthPlugin = (GrowthPlugin*)Simulator::pluginManager.get("Growth");
        if (growthPlugin->FGF(pt.z) > growthPlugin->getFGFThreshold()) active = false;
        
        return ( active && (const_cast<CellG*>(cell)->type == 1) && (concentration > threshold) );
    };
  };
};
#endif
