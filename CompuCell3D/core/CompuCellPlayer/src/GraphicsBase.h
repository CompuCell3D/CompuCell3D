#ifndef _GRAPHICSBASE_H
#define _GRAPHICSBASE_H
#include <vector>

#include <map>
#include <qpen.h>
#include <qbrush.h>
#include <qcolor.h>
#include <qimage.h>

#include <GraphicsDataFields.h>
#include <Utils/Coordinates3D.h>
#include <CompuCell3D/Boundary/BoundaryTypeDefinitions.h>

class QPainter;
class GraphicsDataFields;
class UniversalGraphicsSettings;

class GraphicsBase{

   public:

      typedef GraphicsDataFields::floatField3D_t floatField3D_t;
      typedef std::map<std::string,floatField3D_t*>::iterator floatField3DNameMapItr_t;
      typedef GraphicsDataFields::floatField2D_t floatField2D_t;
      typedef GraphicsDataFields::vectorFloatField3D_t vectorFloatField3D_t;
      typedef std::map<std::string,vectorFloatField3D_t*>::iterator vectorFloatField3DNameMapItr_t;
      typedef GraphicsDataFields::vectorCellFloatField3D_t vectorCellFloatField3D_t;
      typedef std::map<std::string,vectorCellFloatField3D_t*>::iterator vectorCellFloatField3DNameMapItr_t;
      typedef GraphicsDataFields::vectorFieldCellLevel_t vectorFieldCellLevel_t;
      typedef GraphicsDataFields::field3DGraphicsData_t field3DGraphicsData_t;


      typedef void (GraphicsBase::*paintingFcnPtr_t)(void) ;
      
      GraphicsBase();
      virtual ~GraphicsBase();
           
      virtual void paintLattice();
      virtual void paintLegend( float minConcentration, float maxConcentration,std::string location,std::string type);
      virtual unsigned int legendDimension(std::string location, unsigned int &rectWidth,unsigned int & rectHeight,std::string type);
      virtual void paintConcentrationLattice();
      virtual void paintCellVectorFieldLattice(){};
      virtual void fillFakeConcentration();
      virtual void doContourLines();
      virtual void produceImage(QImage & image);
      virtual void produceImage(const std::string & _fileName){};

      paintingFcnPtr_t getCurrentPainitgFcnPtr();
      paintingFcnPtr_t getPaintConcentrationLattice();
      paintingFcnPtr_t getPaintCellVectorFieldLattice();
      paintingFcnPtr_t getPaintLattice();
      void setCurrentPainitgFcnPtr(paintingFcnPtr_t _paintingFcnPtr);
      void drawCurrentScene();
      
      //Setters, getters
      std::map<unsigned short,QPen> & getTypePenMap();
      std::map<unsigned short,QBrush> & getTypeBrushMap();

      void setTypePenMapPtr(std::map<unsigned short,QPen> * mapPtr);
      void setTypeBrushMapPtr(std::map<unsigned short,QBrush> * mapPtr);

      void setDefaultPenPtr(QPen *penPtr);
      void setDefaultBrushPtr(QBrush *brushPtr);

      void setBorderPenPtr(QPen *penPtr);
      void setContourPenPtr(QPen *penPtr);

      void setDefaultColorPtr(QColor *colorPtr);
      void setBorderColorPtr(QColor *colorPtr);
      void setContourColorPtr(QColor *colorPtr);
      void setArrowColorPtr(QColor *colorPtr);
      QColor * getArrowColorPtr();

      
      void setCurrentConcentrationFieldPtr(floatField3D_t *_currentConcentrationFieldPtr);
      void setCurrentVectorFieldPtr(vectorFloatField3D_t *_currentVectorFieldPtr);
      void setCurrentVectorCellFieldPtr(vectorCellFloatField3D_t *_currentVectorCellFieldPtr);
      void setCurrentVectorCellLevelFieldPtr(vectorFieldCellLevel_t *_currentVectorFieldCellLevelPtr);
      
      void toggleMaxConcentration();

      void setMaxConcentrationFixed(bool fixed);
      void setMinConcentrationFixed(bool fixed);
      void setMaxConcentration(float conc);
      void setMinConcentration(float conc);
      float getMaxConcentration();
      float getMinConcentration();
      float getMaxConcentrationTrue();
      float getMinConcentrationTrue();

      bool getMaxConcentrationFixed();
      bool getMinConcentrationFixed();


      void setMaxMagnitudeFixed(bool fixed);
      void setMinMagnitudeFixed(bool fixed);
      void setMaxMagnitude(float conc);
      void setMinMagnitude(float conc);
      void setArrowLength(int _len);
      float getMaxMagnitude();
      float getMinMagnitude();
      float getMaxMagnitudeTrue();
      float getMinMagnitudeTrue();

      bool getMaxMagnitudeFixed();
      bool getMinMagnitudeFixed();
      int getArrowLength();


      bool getOverlayVectorCellFields();
      bool getScaleArrows();
      bool getFixedArrowColorFlag();

      bool setOverlayVectorCellFields(bool _overlayVectorCellFields);
      bool setScaleArrows(bool _scaleArrows);
      bool setFixedArrowColor(bool _fixedArrowColorFlag);

      void setSilentMode(bool _silentMode);
      bool getSilentMode();
      void setXServerFlag(bool _xServerFlag);
      bool getXServerFlag();


      void setGraphicsDataFieldPtr(GraphicsDataFields * _graphFieldsPtr);

      GraphicsDataFields * getGraphFieldsPtr();;
      void setUnivGraphSetPtr(UniversalGraphicsSettings * _univGraphSetPtr);


      void setNumberOfLegendBoxes(unsigned int _n);
      unsigned int getNumberOfLegendBoxes();
      
      void setNumberAccuracy(unsigned int _n);
      unsigned int getNumberAccuracy();
      bool getLegendEnable();
      void setLegendEnable(bool);

      void setNumberOfLegendBoxesVector(unsigned int _n);
      unsigned int getNumberOfLegendBoxesVector();
      
      void setNumberAccuracyVector(unsigned int _n);
      unsigned int getNumberAccuracyVector();
      bool getLegendEnableVector();
      void setLegendEnableVector(bool);
      void setId(QString  _id);
      QString  getId() const;
      void setLatticeType(CompuCell3D::LatticeType _latticeType){latticeType=_latticeType;}
                  
   protected:
		CompuCell3D::LatticeType latticeType;

		paintingFcnPtr_t paintingFcnPtr;

		float percentageQuantity(float _mag,float _min, float _max);

		floatField3D_t *currentConcentrationFieldPtr;

		std::map<unsigned short,QColor> *typeColorMapPtr;
		QColor *defaultColorPtr;
		QColor *borderColorPtr;
		QColor *contourColorPtr;
		QColor *arrowColorPtr;

		vectorFloatField3D_t *currentVectorFieldPtr;
		vectorCellFloatField3D_t *currentVectorCellFieldPtr;

		vectorFieldCellLevel_t *currentVectorCellLevelFieldPtr;

       
		float maxConcentration;
		float minConcentration;
		float maxConcentrationTrue;
		float minConcentrationTrue;


		bool maxConcentrationToggled;
		bool maxConcentrationFixed;
		bool minConcentrationFixed;
		unsigned int numberOfContours;

		float maxMagnitude;
		float minMagnitude;
		float maxMagnitudeTrue;
		float minMagnitudeTrue;


		bool maxMagnitudeToggled;
		bool maxMagnitudeFixed;
		bool minMagnitudeFixed;

				     

		unsigned int legendWidth;
		bool legendEnable;
		 
		unsigned int numberOfLegendBoxes;
		unsigned int numberAccuracy;

		unsigned int numberOfLegendBoxesVector;
		unsigned int numberAccuracyVector;
		bool legendEnableVector;
		int arrowLength;
		bool overlayVectorCellFields;
		bool scaleArrows;
		bool fixedArrowColorFlag;
		bool silentMode;
		bool xServerFlag;


		GraphicsDataFields * graphFieldsPtr;
		UniversalGraphicsSettings *univGraphSetPtr;
		unsigned int L3D;
		unsigned int M3D;
		unsigned int N3D;
		QString id;
};


inline void GraphicsBase::setDefaultColorPtr(QColor *colorPtr){defaultColorPtr=colorPtr;}
inline void GraphicsBase::setBorderColorPtr(QColor *colorPtr){borderColorPtr=colorPtr;}
inline void GraphicsBase::setContourColorPtr(QColor *colorPtr){contourColorPtr=colorPtr;}

inline void GraphicsBase::setArrowColorPtr(QColor *colorPtr){arrowColorPtr=colorPtr;}

inline QColor * GraphicsBase::getArrowColorPtr(){return arrowColorPtr;}

inline void GraphicsBase::setCurrentConcentrationFieldPtr(floatField3D_t *_currentConcentrationFieldPtr)
{
   currentConcentrationFieldPtr=_currentConcentrationFieldPtr;
}

inline void GraphicsBase::setCurrentVectorFieldPtr(vectorFloatField3D_t *_currentVectorFieldPtr)
{
   currentVectorFieldPtr=_currentVectorFieldPtr;
}

inline void GraphicsBase::setCurrentVectorCellFieldPtr(vectorCellFloatField3D_t *_currentVectorCellFieldPtr)
{
   currentVectorCellFieldPtr=_currentVectorCellFieldPtr;
}


inline void GraphicsBase::setCurrentVectorCellLevelFieldPtr(vectorFieldCellLevel_t *_currentVectorFieldCellLevelPtr){
   currentVectorCellLevelFieldPtr=_currentVectorFieldCellLevelPtr;
}



inline void GraphicsBase::toggleMaxConcentration(){maxConcentrationToggled=true;}

inline void GraphicsBase::setMaxConcentrationFixed(bool fixed){maxConcentrationFixed=fixed;}
inline void GraphicsBase::setMinConcentrationFixed(bool fixed){minConcentrationFixed=fixed;}
inline void GraphicsBase::setMaxConcentration(float conc){maxConcentration = conc;}
inline void GraphicsBase::setMinConcentration(float conc){minConcentration = conc;}
inline float GraphicsBase::getMaxConcentration(){return maxConcentration;}
inline float GraphicsBase::getMinConcentration(){return minConcentration;}
inline float GraphicsBase::getMaxConcentrationTrue(){return maxConcentrationTrue;}
inline float GraphicsBase::getMinConcentrationTrue(){return minConcentrationTrue;}

inline bool GraphicsBase::getMaxConcentrationFixed(){return maxConcentrationFixed;}
inline bool GraphicsBase::getMinConcentrationFixed(){return minConcentrationFixed;}

inline void GraphicsBase::setNumberOfLegendBoxes(unsigned int _n){numberOfLegendBoxes=_n;}
inline unsigned int GraphicsBase::getNumberOfLegendBoxes(){return numberOfLegendBoxes;}

inline void GraphicsBase::setNumberAccuracy(unsigned int _n){numberAccuracy=_n;}
inline unsigned int GraphicsBase::getNumberAccuracy(){return numberAccuracy;}

inline bool GraphicsBase::getLegendEnable(){return legendEnable;}
inline void GraphicsBase::setLegendEnable(bool _flag){legendEnable=_flag;}


inline void GraphicsBase::setNumberOfLegendBoxesVector(unsigned int _n){numberOfLegendBoxesVector=_n;}
inline unsigned int GraphicsBase::getNumberOfLegendBoxesVector(){return numberOfLegendBoxesVector;}

inline void GraphicsBase::setNumberAccuracyVector(unsigned int _n){numberAccuracyVector=_n;}
inline unsigned int GraphicsBase::getNumberAccuracyVector(){return numberAccuracyVector;}

inline bool GraphicsBase::getLegendEnableVector(){return legendEnableVector;}
inline void GraphicsBase::setLegendEnableVector(bool _flag){legendEnableVector=_flag;}



inline void GraphicsBase::setMaxMagnitudeFixed(bool fixed){maxMagnitudeFixed=fixed;}
inline void GraphicsBase::setMinMagnitudeFixed(bool fixed){minMagnitudeFixed=fixed;}
inline void GraphicsBase::setMaxMagnitude(float conc){maxMagnitude = conc;}
inline void GraphicsBase::setMinMagnitude(float conc){minMagnitude = conc;}
inline void GraphicsBase::setArrowLength(int _len){arrowLength=_len;}
inline float GraphicsBase::getMaxMagnitude(){return maxMagnitude;}
inline float GraphicsBase::getMinMagnitude(){return minMagnitude;}
inline float GraphicsBase::getMaxMagnitudeTrue(){return maxMagnitudeTrue;}
inline float GraphicsBase::getMinMagnitudeTrue(){return minMagnitudeTrue;}

inline bool GraphicsBase::getMaxMagnitudeFixed(){return maxMagnitudeFixed;}
inline bool GraphicsBase::getMinMagnitudeFixed(){return minMagnitudeFixed;}
inline int GraphicsBase::getArrowLength(){return arrowLength;}

inline bool GraphicsBase::getOverlayVectorCellFields(){return overlayVectorCellFields;}
inline bool GraphicsBase::getScaleArrows(){return scaleArrows;}
inline bool GraphicsBase::getFixedArrowColorFlag(){return fixedArrowColorFlag;}

inline void GraphicsBase::setSilentMode(bool _silentMode){silentMode=_silentMode;}
inline bool GraphicsBase::getSilentMode(){return silentMode;}

inline void GraphicsBase::setXServerFlag(bool _xServerFlag){xServerFlag=_xServerFlag;}
inline bool GraphicsBase::getXServerFlag(){return xServerFlag;}


inline bool GraphicsBase::setOverlayVectorCellFields(bool _overlayVectorCellFields)
{overlayVectorCellFields=_overlayVectorCellFields; return true;}

inline bool GraphicsBase::setScaleArrows(bool _scaleArrows){scaleArrows=_scaleArrows; return true;}
inline bool GraphicsBase::setFixedArrowColor(bool _fixedArrowColorFlag){fixedArrowColorFlag=_fixedArrowColorFlag; return true;}


      
inline void GraphicsBase::setGraphicsDataFieldPtr(GraphicsDataFields * _graphFieldsPtr){
   graphFieldsPtr=_graphFieldsPtr;
}




inline GraphicsDataFields * GraphicsBase::getGraphFieldsPtr(){return graphFieldsPtr;};
inline void GraphicsBase::setUnivGraphSetPtr(UniversalGraphicsSettings * _univGraphSetPtr){univGraphSetPtr=_univGraphSetPtr;}

inline GraphicsBase::paintingFcnPtr_t GraphicsBase::getPaintConcentrationLattice(){return &GraphicsBase::paintConcentrationLattice;}
inline GraphicsBase::paintingFcnPtr_t GraphicsBase::getPaintCellVectorFieldLattice(){return &GraphicsBase::paintCellVectorFieldLattice;}
inline GraphicsBase::paintingFcnPtr_t GraphicsBase::getPaintLattice(){return &GraphicsBase::paintLattice;}



// inline void GraphicsBase::drawCurrentScene(){
//    (this->*paintingFcnPtr)();
// }


inline void GraphicsBase::setId(QString  _id){id=_id;}
inline QString  GraphicsBase::getId() const{return id;}


class VectorFieldPlotItem{
   public:
      VectorFieldPlotItem():
      CM(Coordinates3D<int>(0,0,0)),
      vectorQuant(Coordinates3D<float>(0.,0.,0.)),
      dimVolume(0)
      {}
      VectorFieldPlotItem(const Coordinates3D<int> &_CM,int _vol, const Coordinates3D<float> & _vecQuant):
      CM(Coordinates3D<int>(_CM)),
      vectorQuant(Coordinates3D<float>(_vecQuant)),
      dimVolume(_vol)
      {}
      
      Coordinates3D<int> CM;//Center of Mass or centroid for 2D case
      Coordinates3D<float> vectorQuant;//vector quantity
      int dimVolume;//can be either volume for 3D or surface dor 2D cases
};




#endif
