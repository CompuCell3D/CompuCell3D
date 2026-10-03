#include "FocalPointPlasticityPlugin.h"
#include <cc3d/CompiledSteppablePluginAPI.h>
#include <vector>
namespace {
    CompuCell3D::FocalPointPlasticityPlugin *p(void *x) { return static_cast<CompuCell3D::FocalPointPlasticityPlugin *>(x); }

    struct LinkIterator { std::vector<CompuCell3D::FocalPointPlasticityLinkBase *> values; size_t index = 0; };
    struct CellIterator { std::vector<CompuCell3D::CellG *> values; size_t index = 0; };

    CC3DCellViewV1 cellView(CompuCell3D::CellG *cell) {
        CC3DCellViewV1 view{};
        if (!cell) return view;
        view.handle = cell;
        view.id = &cell->id; view.clusterId = &cell->clusterId; view.volume = &cell->volume;
        view.surface = &cell->surface; view.clusterSurface = &cell->clusterSurface;
        view.type = &cell->type; view.subtype = &cell->subtype;
        view.targetVolume = &cell->targetVolume; view.lambdaVolume = &cell->lambdaVolume;
        view.targetSurface = &cell->targetSurface; view.angle = &cell->angle;
        view.lambdaSurface = &cell->lambdaSurface; view.targetClusterSurface = &cell->targetClusterSurface;
        view.lambdaClusterSurface = &cell->lambdaClusterSurface;
        view.xCM = &cell->xCM; view.yCM = &cell->yCM; view.zCM = &cell->zCM;
        view.xCOM = &cell->xCOM; view.yCOM = &cell->yCOM; view.zCOM = &cell->zCOM;
        view.xCOMPrev = &cell->xCOMPrev; view.yCOMPrev = &cell->yCOMPrev; view.zCOMPrev = &cell->zCOMPrev;
        view.iXX = &cell->iXX; view.iXY = &cell->iXY; view.iXZ = &cell->iXZ;
        view.iYY = &cell->iYY; view.iYZ = &cell->iYZ; view.iZZ = &cell->iZZ;
        view.lX = &cell->lX; view.lY = &cell->lY; view.lZ = &cell->lZ; view.ecc = &cell->ecc;
        view.lambdaVecX = &cell->lambdaVecX; view.lambdaVecY = &cell->lambdaVecY; view.lambdaVecZ = &cell->lambdaVecZ;
        view.flag = &cell->flag; view.averageConcentration = &cell->averageConcentration;
        view.fluctAmpl = &cell->fluctAmpl; view.lambdaMotility = &cell->lambdaMotility;
        view.biasVecX = &cell->biasVecX; view.biasVecY = &cell->biasVecY; view.biasVecZ = &cell->biasVecZ;
        view.connectivityOn = &cell->connectivityOn;
        return view;
    }

    CC3DFPPLinkViewV1 linkView(CompuCell3D::FocalPointPlasticityLinkBase *link) {
        CC3DFPPLinkViewV1 view{};
        if (!link) return view;
        view.handle = link; view.initiator = cellView(link->getObj0()); view.initiated = cellView(link->getObj1());
        view.valid = 1u;
        if (link->isAnchor()) {
            auto *anchor = dynamic_cast<CompuCell3D::FocalPointPlasticityAnchor *>(link);
            view.anchorId = anchor ? anchor->getAnchorId() : -1;
        }
        return view;
    }

    CC3DFPPLinkIteratorHandle linkBegin(void *x, CC3DPluginCellHandle cell, int kind) {
        auto *state = new LinkIterator;
        auto *c = static_cast<CompuCell3D::CellG *>(cell);
        if (kind == 1) { for (auto *v : p(x)->getFPPInternalLinkList(c)) state->values.push_back(v); }
        else if (kind == 2) { for (auto *v : p(x)->getFPPAnchorList(c)) state->values.push_back(v); }
        else { for (auto *v : p(x)->getFPPLinkList(c)) state->values.push_back(v); }
        return state;
    }
    uint8_t linkValid(void *, CC3DFPPLinkIteratorHandle h) { auto *s = static_cast<LinkIterator *>(h); return s && s->index < s->values.size(); }
    CC3DFPPLinkViewV1 linkGet(void *, CC3DFPPLinkIteratorHandle h) { auto *s = static_cast<LinkIterator *>(h); return s && s->index < s->values.size() ? linkView(s->values[s->index]) : CC3DFPPLinkViewV1{}; }
    void linkNext(void *, CC3DFPPLinkIteratorHandle h) { auto *s = static_cast<LinkIterator *>(h); if (s && s->index < s->values.size()) ++s->index; }
    void linkDestroy(void *, CC3DFPPLinkIteratorHandle h) { delete static_cast<LinkIterator *>(h); }
    CC3DFPPLinkViewV1 findLink(void *x, CC3DPluginCellHandle a, CC3DPluginCellHandle b, bool internal) { return linkView(internal ? static_cast<CompuCell3D::FocalPointPlasticityLinkBase *>(p(x)->getFPPInternalLinkByCells(static_cast<CompuCell3D::CellG *>(a), static_cast<CompuCell3D::CellG *>(b))) : static_cast<CompuCell3D::FocalPointPlasticityLinkBase *>(p(x)->getFPPLinkByCells(static_cast<CompuCell3D::CellG *>(a), static_cast<CompuCell3D::CellG *>(b)))); }
    size_t linkCount(void *x, CC3DPluginCellHandle c, int kind) { auto *cell = static_cast<CompuCell3D::CellG *>(c); return kind == 1 ? p(x)->getFPPInternalLinkCount(cell) : kind == 2 ? p(x)->getFPPAnchorCount(cell) : p(x)->getFPPLinkCount(cell); }
    CC3DFPPCellIteratorHandle cellBegin(void *x, CC3DPluginCellHandle cell, bool internal) { auto *s = new CellIterator; s->values = p(x)->getFPPLinkedCells(static_cast<CompuCell3D::CellG *>(cell), internal); return s; }
    uint8_t cellValid(void *, CC3DFPPCellIteratorHandle h) { auto *s = static_cast<CellIterator *>(h); return s && s->index < s->values.size(); }
    CC3DCellViewV1 cellGet(void *, CC3DFPPCellIteratorHandle h) { auto *s = static_cast<CellIterator *>(h); return s && s->index < s->values.size() ? cellView(s->values[s->index]) : CC3DCellViewV1{}; }
    void cellNext(void *, CC3DFPPCellIteratorHandle h) { auto *s = static_cast<CellIterator *>(h); if (s && s->index < s->values.size()) ++s->index; }
    void cellDestroy(void *, CC3DFPPCellIteratorHandle h) { delete static_cast<CellIterator *>(h); }
}
extern "C" void cc3d_register_focal_point_plasticity_plugin(void *s, void *x) {
    CC3DFocalPointPlasticityCallbacksV1 c{}; c.plugin_instance=x;
    c.create_link_fn=[](void*x,void*a,void*b,double l,double t,double m){p(x)->createFocalPointPlasticityLink(static_cast<CompuCell3D::CellG*>(a),static_cast<CompuCell3D::CellG*>(b),l,t,m);};
    c.create_internal_link_fn=[](void*x,void*a,void*b,double l,double t,double m){p(x)->createInternalFocalPointPlasticityLink(static_cast<CompuCell3D::CellG*>(a),static_cast<CompuCell3D::CellG*>(b),l,t,m);};
    c.delete_link_fn=[](void*x,void*a,void*b){p(x)->deleteFocalPointPlasticityLink(static_cast<CompuCell3D::CellG*>(a),static_cast<CompuCell3D::CellG*>(b));};
    c.delete_internal_link_fn=[](void*x,void*a,void*b){p(x)->deleteInternalFocalPointPlasticityLink(static_cast<CompuCell3D::CellG*>(a),static_cast<CompuCell3D::CellG*>(b));};
    c.set_link_fn=[](void*x,void*a,void*b,double l,double t,double m){p(x)->setFocalPointPlasticityParameters(static_cast<CompuCell3D::CellG*>(a),static_cast<CompuCell3D::CellG*>(b),l,t,m);};
    c.get_lambda_fn=[](void*x,void*a,void*b){return p(x)->getPlasticityParametersLambdaDistance(static_cast<CompuCell3D::CellG*>(a),static_cast<CompuCell3D::CellG*>(b));};
    c.get_target_fn=[](void*x,void*a,void*b){return p(x)->getPlasticityParametersTargetDistance(static_cast<CompuCell3D::CellG*>(a),static_cast<CompuCell3D::CellG*>(b));};
    c.create_anchor_fn=[](void*x,void*a,double l,double t,double m,float X,float Y,float Z){return p(x)->createAnchor(static_cast<CompuCell3D::CellG*>(a),l,t,m,X,Y,Z);};
    c.delete_anchor_fn=[](void*x,void*a,int i){p(x)->deleteAnchor(static_cast<CompuCell3D::CellG*>(a),i);};
    c.set_anchor_fn=[](void*x,void*a,int i,double l,double t,double m,float X,float Y,float Z){p(x)->setAnchorParameters(static_cast<CompuCell3D::CellG*>(a),i,l,t,m,X,Y,Z);};
    c.links_begin_fn=[](void*x,void*a){return linkBegin(x,a,0);}; c.links_valid_fn=linkValid; c.links_get_fn=linkGet; c.links_next_fn=linkNext; c.links_destroy_fn=linkDestroy;
    c.internal_links_begin_fn=[](void*x,void*a){return linkBegin(x,a,1);}; c.internal_links_valid_fn=linkValid; c.internal_links_get_fn=linkGet; c.internal_links_next_fn=linkNext; c.internal_links_destroy_fn=linkDestroy;
    c.anchors_begin_fn=[](void*x,void*a){return linkBegin(x,a,2);}; c.anchors_valid_fn=linkValid; c.anchors_get_fn=linkGet; c.anchors_next_fn=linkNext; c.anchors_destroy_fn=linkDestroy;
    c.find_link_fn=[](void*x,void*a,void*b){return findLink(x,a,b,false);}; c.find_internal_link_fn=[](void*x,void*a,void*b){return findLink(x,a,b,true);};
    c.link_count_fn=[](void*x,void*a){return linkCount(x,a,0);}; c.internal_link_count_fn=[](void*x,void*a){return linkCount(x,a,1);}; c.anchor_count_fn=[](void*x,void*a){return linkCount(x,a,2);};
    c.linked_cells_begin_fn=[](void*x,void*a){return cellBegin(x,a,false);}; c.linked_cells_valid_fn=cellValid; c.linked_cells_get_fn=cellGet; c.linked_cells_next_fn=cellNext; c.linked_cells_destroy_fn=cellDestroy;
    c.internal_linked_cells_begin_fn=[](void*x,void*a){return cellBegin(x,a,true);}; c.internal_linked_cells_valid_fn=cellValid; c.internal_linked_cells_get_fn=cellGet; c.internal_linked_cells_next_fn=cellNext; c.internal_linked_cells_destroy_fn=cellDestroy;
    cc3d_register_focal_point_plasticity_callbacks(s,&c);
}
