#ifndef _ASTINDEXNAMENODE_H_
#define _ASTINDEXNAMENODE_H_

#include "sbml/math/ASTNode.h"

#include "sbmlsolver/exportdefs.h"

#ifdef __cplusplus

class ASTIndexNameNode :
public ASTNode
{
 public:
  ASTIndexNameNode();
  virtual ~ASTIndexNameNode(void);

  unsigned int getIndex() const { return index; }
  unsigned int isSetIndex() const { return indexSet; }
  void setIndex(unsigned int i) { index = i; indexSet = 1; }

  unsigned int isSetData() const { return dataSet; }
  void setData() { dataSet = 1; } 

 private:
  unsigned int index ;
  int indexSet ;
 
  /* used for differentiating whether the name refers
     to a variable for which "data" is available   */
  int dataSet;
};

#endif /* __cplusplus */

BEGIN_C_DECLS

/* creates a new AST node with an index field */
SBML_ODESOLVER_API ASTNode_t *ASTNode_createIndexName(void);

/* returns 1 if the node is indexed */
SBML_ODESOLVER_API int ASTNode_isIndexName(ASTNode_t *);

/* assumes node is index node */ 
SBML_ODESOLVER_API unsigned int ASTNode_getIndex(ASTNode_t *); 

/* returns 0 if node isn't index or if index is not set yet */
SBML_ODESOLVER_API unsigned int ASTNode_isSetIndex(ASTNode_t *);

/* assumes node is index node */
SBML_ODESOLVER_API void ASTNode_setIndex(ASTNode_t *, unsigned int); 

/* returns 0 if node isn't index or if data is not set yet */
SBML_ODESOLVER_API unsigned int ASTNode_isSetData(ASTNode_t *);

/* assumes node is index node, and then sets data  */
SBML_ODESOLVER_API void ASTNode_setData(ASTNode_t *); 




END_C_DECLS

#endif /* _ASTINDEXNAMENODE_H_ */
