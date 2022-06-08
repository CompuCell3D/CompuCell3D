#ifndef FOCALPOINTPLASTICITYLINKINVENTORY_H
#define FOCALPOINTPLASTICITYLINKINVENTORY_H

#include "FocalPointPlasticityLinkInventoryBase.h"

namespace CompuCell3D {

    /**
    Written by T.J. Sego, Ph.D.
    */

    class FocalPointPlasticityLink;

    class FocalPointPlasticityInternalLink;

    class FocalPointPlasticityAnchor;

    class FOCALPOINTPLASTICITY_EXPORT FPPLinkInventory : public FPPLinkInventoryBase<FocalPointPlasticityLink> {

    public:

        FPPLinkInventory() : FPPLinkInventoryBase() {}

        FPPLinkInventory(
                ExtraMembersGroupAccessor <
                FPPLinkInventoryTracker<FocalPointPlasticityLink>> *_cellLinkInventoryTrackerAccessor,
                Potts3D *_potts) : FPPLinkInventoryBase(_cellLinkInventoryTrackerAccessor, _potts) {

        }

        virtual ~FPPLinkInventory() {}

        // Get the link connecting two cells
        FocalPointPlasticityLink *getLinkByCells(CellG *_cell0, CellG *_cell1) {
            return getLinkById(FPPLinkID(_cell0->id, _cell1->id));
        }

        // Get list of cells linked to a cell
        std::vector<CellG *> getLinkedCells(CellG *_cell) {
            std::vector < CellG * > o;
            FPPInventory_t *cInv = getCellLinkInventory(_cell);
            for (linkInventoryItr_t itr = cInv->linkInventoryBegin(); itr != cInv->linkInventoryEnd(); ++itr) {
                o.push_back(itr->second->getOtherCell(_cell));
            }
            return o;
        }

        // Get number of junctions for a cell by type
        int getNumberOfJunctionsByType(CellG *_cell, unsigned char _type) {
            FPPInventory_t *cInv = getCellLinkInventory(_cell);
            return count_if(cInv->linkInventoryBegin(), cInv->linkInventoryEnd(), [&](linkInventoryPair_t p) {
                return p.second->getOtherCell(_cell)->type == _type;
            });
        }

    };

    class FOCALPOINTPLASTICITY_EXPORT FPPInternalLinkInventory
            : public FPPLinkInventoryBase<FocalPointPlasticityInternalLink> {

    public:

        FPPInternalLinkInventory() : FPPLinkInventoryBase() {}

        FPPInternalLinkInventory(
                ExtraMembersGroupAccessor <
                FPPLinkInventoryTracker<FocalPointPlasticityInternalLink>> *_cellLinkInventoryTrackerAccessor,
                Potts3D *_potts) : FPPLinkInventoryBase(_cellLinkInventoryTrackerAccessor, _potts) {

        }

        virtual ~FPPInternalLinkInventory() {}

        // Get the link connecting two cells
        FocalPointPlasticityInternalLink *getLinkByCells(CellG *_cell0, CellG *_cell1) {
            return getLinkById(FPPLinkID(_cell0->id, _cell1->id));
        }

        // Get list of cells linked to a cell
        std::vector<CellG *> getLinkedCells(CellG *_cell) {
            std::vector < CellG * > o;
            FPPInventory_t *cInv = getCellLinkInventory(_cell);
            for (linkInventoryItr_t itr = cInv->linkInventoryBegin(); itr != cInv->linkInventoryEnd(); ++itr) {
                o.push_back(itr->second->getOtherCell(_cell));
            }
            return o;
        }

        // Get number of junctions for a cell by type
        int getNumberOfJunctionsByType(CellG *_cell, unsigned char _type) {
            FPPInventory_t *cInv = getCellLinkInventory(_cell);
            return count_if(cInv->linkInventoryBegin(), cInv->linkInventoryEnd(), [&](linkInventoryPair_t p) {
                return p.second->getOtherCell(_cell)->type == _type;
            });
        }

    };

    class FOCALPOINTPLASTICITY_EXPORT FPPAnchorInventory : public FPPLinkInventoryBase<FocalPointPlasticityAnchor> {

    public:

        FPPAnchorInventory() : FPPLinkInventoryBase() {}

        FPPAnchorInventory(
                ExtraMembersGroupAccessor <
                FPPLinkInventoryTracker<FocalPointPlasticityAnchor>> *_cellLinkInventoryTrackerAccessor,
                Potts3D *_potts) : FPPLinkInventoryBase(_cellLinkInventoryTrackerAccessor, _potts) {

        }

        virtual ~FPPAnchorInventory() {}

        FocalPointPlasticityAnchor *getAnchor(CellG *_cell, long _anchorId) {
            return getLinkById(FPPLinkID(_cell->id, _anchorId));
        }

        int getNextAnchorId(CellG *_cell) {
            FPPLinkList ll = getCellLinkList(_cell);
            if (ll.size() == 0) return int(0);
            else return (*ll.end())->getAnchorId() + 1;
        }

    };

}

#endif