
#ifndef CACELL_H
#define CACELL_H


#ifndef PyObject_HEAD
struct _object; //forward declare
typedef _object PyObject; //type redefinition
#endif

class BasicClassGroup;

namespace CompuCell3D {

  /**
   * A Potts3D cell.
   */

   class CACell{
   public:
      typedef unsigned char CellType_t;
      CACell():
        type(0),

		xCOM(-1),yCOM(-1),zCOM(-1),
		xCOMPrev(-1),yCOMPrev(-1),zCOMPrev(-1),
        id(0),
        clusterId(0),
		size(1),
        extraAttribPtr(0),
        pyAttrib(0)
      {}
      unsigned char type;
	  long xCOM,yCOM,zCOM; // numerator of center of mass expression (components)
	  long xCOMPrev,yCOMPrev,zCOMPrev; // previous center of mass 
      long id; //id of a cell
      long clusterId; //clusterId to which cell belongs - for now notused but I keep it just in case
	  int size; //size of the cell
      BasicClassGroup *extraAttribPtr;

      PyObject *pyAttrib;
   };

};
#endif
