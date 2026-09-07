#include "FocalPointPlasticityPlugin.h"
#include <cc3d/CompiledSteppablePluginAPI.h>
namespace { CompuCell3D::FocalPointPlasticityPlugin *p(void*x){return static_cast<CompuCell3D::FocalPointPlasticityPlugin*>(x);} }
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
    cc3d_register_focal_point_plasticity_callbacks(s,&c);
}
