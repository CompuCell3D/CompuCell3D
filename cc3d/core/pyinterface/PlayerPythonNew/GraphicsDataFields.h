#ifndef GRAPHICSDATAFIELDS_H
#define GRAPHICSDATAFIELDS_H
#include <vector>
#include <map>
#include <string>
#include <Utils/Coordinates3D.h>
#include "GraphicsData.h"

#include <CompuCell3D/Potts3D/Cell.h>
#include "PlayerPythonDLLSpecifier.h"

class PLAYERPYTHONNEW_EXPORT GraphicsDataFields{
   public://typedef's
      typedef std::vector<std::vector<float> > floatField2D_t;
      typedef std::vector<std::vector<std::vector<float> > > floatField3D_t;
      typedef std::map<std::string,floatField3D_t*>::iterator floatField3DNameMapItr_t;
      typedef std::vector<std::vector<std::vector<Coordinates3D<float> > > > vectorFloatField3D_t;
      typedef std::map<std::string,vectorFloatField3D_t*>::iterator vectorFloatField3DNameMapItr_t;
      typedef std::vector<std::vector<std::vector<std::pair<Coordinates3D<float>,CompuCell3D::CellG*> > > > vectorCellFloatField3D_t;
      typedef std::map<std::string,vectorCellFloatField3D_t*>::iterator vectorCellFloatField3DNameMapItr_t;

      //cell level vector fields (represented as maps)
      typedef std::map<CompuCell3D::CellG*,Coordinates3D<float> > vectorFieldCellLevel_t;
      typedef std::map<std::string,vectorFieldCellLevel_t *>::iterator vectorFieldCellLevelNameMapItr_t;
      typedef vectorFieldCellLevel_t::iterator vectorFieldCellLevelItr_t;

      typedef std::vector<std::vector<std::vector<GraphicsData> > > field3DGraphicsData_t;

      typedef std::map<std::string,std::string> plotNamePlotTypeMap_t;
      //The above map is ued to mark what type a give plot is of For example cAMP will be of type "scalar"
      //The allowed key_words are
      //cell_field - for plots of te cell types
      //scalar - for concentration plots
      //vector_cell_level - for vector plots where vector is an attribute of a cell
      //vector_pixel_level - for vector plots where vector is an attribute of a pixel
   public:

      GraphicsDataFields();
      ~GraphicsDataFields();

      unsigned int getSizeL(){return sizeL;}
      unsigned int getSizeM(){return sizeM;}
      unsigned int getSizeN(){return sizeN;}

      void allocateField3D(unsigned int _sizeL, unsigned int _sizeM, unsigned int _sizeN);
      void allocateFloatField3D(unsigned int _sizeL, unsigned int _sizeM, unsigned int _sizeN,std::string _name);
      void allocateVectorFloatField3D(unsigned int _sizeL, unsigned int _sizeM, unsigned int _sizeN,std::string _name);
      void allocateVectorCellFloatField3D(unsigned int _sizeL, unsigned int _sizeM, unsigned int _sizeN,std::string _name);
      void allocateVectorFieldCellLevel(std::string _name);
      void clearAllocatedFields();

      std::vector<std::vector<std::vector<GraphicsData> > > field3DGraphicsData;/// 3d field
      floatField3D_t field3DConcentration;/// 3d field of floats
      std::map<std::string,floatField3D_t*> & getFloatField3DNameMap(){return floatField3DNameMap;}
      std::map<std::string,vectorFloatField3D_t*> & getVectorFloatField3DNameMap(){return vectorFloatField3DNameMap;}
      std::map<std::string,vectorCellFloatField3D_t*> & getVectorCellFloatField3DNameMap(){return vectorCellFloatField3DNameMap;}

      std::map<std::string,vectorFieldCellLevel_t *> & getVectorFieldCellLevelNameMap(){return vectorFieldCellLevelNameMap;}


      plotNamePlotTypeMap_t & getPlotNameplotTypeMap(){return plotNamePlotTypeMap;}

      void insertPlotNamePlotTypePair(const std::string & _plotName , const std::string & _plotType);
      std::string checkPlotType(const std::string & _plotName);
   private:

      unsigned int sizeL, sizeM, sizeN;
      std::map<std::string,floatField3D_t*> floatField3DNameMap; ///this map will be filled externally
      std::map<std::string,vectorFloatField3D_t*> vectorFloatField3DNameMap; ///this map will be filled externally
      std::map<std::string,vectorCellFloatField3D_t*> vectorCellFloatField3DNameMap; ///this map will be filled externally

      std::map<std::string, vectorFieldCellLevel_t * > vectorFieldCellLevelNameMap;///this map will be filled externally

      plotNamePlotTypeMap_t plotNamePlotTypeMap;///this map will be filled externally
};






#endif
