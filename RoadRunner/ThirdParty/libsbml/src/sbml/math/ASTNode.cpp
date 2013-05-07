/**
 * @file    ASTNode.cpp
 * @brief   Abstract Syntax Tree (AST) for representing formula trees.
 * @author  Ben Bornstein
 * 
 * <!--------------------------------------------------------------------------
 * This file is part of libSBML.  Please visit http://sbml.org for more
 * information about SBML, and the latest version of libSBML.
 *
 * Copyright (C) 2009-2012 jointly by the following organizations: 
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. EMBL European Bioinformatics Institute (EBML-EBI), Hinxton, UK
 *  
 * Copyright (C) 2006-2008 by the California Institute of Technology,
 *     Pasadena, CA, USA 
 *  
 * Copyright (C) 2002-2005 jointly by the following organizations: 
 *     1. California Institute of Technology, Pasadena, CA, USA
 *     2. Japan Science and Technology Agency, Japan
 * 
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation.  A copy of the license agreement is provided
 * in the file named "LICENSE.txt" included with this software distribution and
 * also available online as http://sbml.org/software/libsbml/license.html
 * ---------------------------------------------------------------------- -->*/

#include <new>
#include <cmath>
#include <stdlib.h>
#include <limits.h>

#include <sbml/common/common.h>
#include <sbml/util/List.h>

#include <sbml/math/ASTNode.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/xml/XMLNode.h>
#include <sbml/Model.h>

/** @cond doxygen-ignored */

using namespace std;

/** @endcond */

LIBSBML_CPP_NAMESPACE_BEGIN

/**
 * ASTNodeType predicates
 */
#define ASTNodeType_isAvogadro(t) \
  (t == AST_NAME_AVOGADRO)

#define ASTNodeType_isConstant(t) \
  (((t >= AST_CONSTANT_E) && (t <= AST_CONSTANT_TRUE)) || t == AST_NAME_AVOGADRO)

#define ASTNodeType_isFunction(t) \
  ((t >= AST_FUNCTION) && (t <= AST_FUNCTION_TANH))

#define ASTNodeType_isLambda(t) \
  (t == AST_LAMBDA)

#define ASTNodeType_isLogical(t) \
  ((t >= AST_LOGICAL_AND) && (t <= AST_LOGICAL_XOR))

#define ASTNodeType_isName(t) \
  ((t >= AST_NAME) && (t <= AST_NAME_TIME))

#define ASTNodeType_isRelational(t) \
  ((t >= AST_RELATIONAL_EQ) && (t <= AST_RELATIONAL_NEQ))

#define ASTNodeType_isInteger(t) \
  (t == AST_INTEGER)

#define ASTNodeType_isRational(t) \
  (t == AST_RATIONAL)

#define ASTNodeType_isReal(t) \
  ((t >= AST_REAL) && (t <= AST_RATIONAL))

#define ASTNodeType_isNumber(t) \
  (ASTNodeType_isInteger(t) || ASTNodeType_isReal(t))

#define ASTNodeType_isOperator(t) \
  ( ( t == AST_PLUS   ) || \
    ( t == AST_MINUS  ) || \
    ( t == AST_TIMES  ) || \
    ( t == AST_DIVIDE ) || \
    ( t == AST_POWER  ) )

#define ASTNodeType_isUnknown(t) \
  (t == AST_UNKNOWN)


/**
 * String Constants
 */
static const char *AST_LAMBDA_STRING = "lambda";

static const char *AST_CONSTANT_STRINGS[] =
{
    "exponentiale"
  , "false"
  , "pi"
  , "true"
};

static const char *AST_FUNCTION_STRINGS[] =
{
    "abs"
  , "arccos"
  , "arccosh"
  , "arccot"
  , "arccoth"
  , "arccsc"
  , "arccsch"
  , "arcsec"
  , "arcsech"
  , "arcsin"
  , "arcsinh"
  , "arctan"
  , "arctanh"
  , "ceiling"
  , "cos"
  , "cosh"
  , "cot"
  , "coth"
  , "csc"
  , "csch"
  , "delay"
  , "exp"
  , "factorial"
  , "floor"
  , "ln"
  , "log"
  , "piecewise"
  , "power"
  , "root"
  , "sec"
  , "sech"
  , "sin"
  , "sinh"
  , "tan"
  , "tanh"
};


static const char *AST_LOGICAL_STRINGS[] =
{
    "and"
  , "not"
  , "or"
  , "xor"
};


static const char *AST_RELATIONAL_STRINGS[] =
{
    "eq"
  , "geq"
  , "gt"
  , "leq"
  , "lt"
  , "neq"
};


static const char *AST_OPERATOR_STRINGS[] =
{
    "divide"
  , "minus"
  , "plus"
  , "times"
  , "power"
};


/*
 * Creates a new ASTNode.
 *
 * By default, node will have a type of AST_UNKNOWN and should be set to
 * something else as soon as possible.
 */
LIBSBML_EXTERN
ASTNode::ASTNode (ASTNodeType_t type)
{
  unsetSemanticsFlag();
  mDefinitionURL = new XMLAttributes();
  mReal          = 0;
  mExponent      = 0;
  mType          = AST_UNKNOWN;
  mChar          = 0;
  mName          = NULL;
  mInteger       = 0;
  mDenominator   = 1;
  mParentSBMLObject = NULL;
  mUnits         = "";
  mId			 = "";
  mClass		 = "";
  mStyle		 = "";
  mIsBvar = false;
  mUserData      = NULL;

  setType(type);

  mChildren             = new List;
  mSemanticsAnnotations = new List;
}


/*
 * Creates a new ASTNode from the given Token_t structure.                   
 *                                                                           
 * The resulting ASTNode will contain the same data as the Token_t           
 * object.  Please refer to the documentation for Token_t to learn           
 * about the possible contents.                                              
 */
LIBSBML_EXTERN
ASTNode::ASTNode (Token_t* token)
{
  unsetSemanticsFlag();
  mDefinitionURL = new XMLAttributes();
  mReal          = 0;
  mExponent      = 0;
  mType          = AST_UNKNOWN;
  mChar          = 0;
  mName          = NULL;
  mInteger       = 0;
  mDenominator   = 1;
  mParentSBMLObject = NULL;
  mUnits         = "";
  mId			 = "";
  mClass		 = "";
  mStyle		 = "";
  mIsBvar = false;
  mUserData      = NULL;

  mChildren             = new List;
  mSemanticsAnnotations = new List;

  if (token != NULL)
  {
    if (token->type == TT_NAME)
    {
      setName(token->value.name);
    }
    else if (token->type == TT_INTEGER)
    {
      setValue(token->value.integer);
    }
    else if (token->type == TT_REAL)
    {
      setValue(token->value.real);
    }
    else if (token->type == TT_REAL_E)
    {
      setValue(token->value.real, token->exponent);
    }
    else
    {
      setCharacter(token->value.ch);
    }
  }
}

/*
 * 
 * Copy constructor; Creates a deep copy of the given ASTNode
 *
 */
LIBSBML_EXTERN
ASTNode::ASTNode (const ASTNode& orig) :
  mType                 ( orig.mType )
 ,mChar                 ( orig.mChar )
 ,mName                 ( NULL )
 ,mInteger              ( orig.mInteger )
 ,mReal                 ( orig.mReal )
 ,mDenominator          ( orig.mDenominator )
 ,mExponent             ( orig.mExponent )
 ,mDefinitionURL        ( orig.mDefinitionURL->clone() )	
 ,hasSemantics          ( orig.hasSemantics )
 ,mChildren             ( new List() )
 ,mSemanticsAnnotations ( new List() )
 ,mParentSBMLObject     ( orig.mParentSBMLObject )
 ,mUnits                ( orig.mUnits)
 ,mId                   ( orig.mId)
 ,mClass                ( orig.mClass)
 ,mStyle                ( orig.mStyle)
 ,mIsBvar               ( orig.mIsBvar)
 ,mUserData             ( orig.mUserData )
{
  if (orig.mName)
  {
    mName = safe_strdup(orig.mName);
  }

  for (unsigned int c = 0; c < orig.getNumChildren(); ++c)
  {
    addChild( orig.getChild(c)->deepCopy() );
  }

  for (unsigned int c = 0; c < orig.getNumSemanticsAnnotations(); ++c)
  {
    addSemanticsAnnotation( orig.getSemanticsAnnotation(c)->clone() );
  }
}

/*
 * 
 * assignment operator
 *
 */
LIBSBML_EXTERN
ASTNode& ASTNode::operator=(const ASTNode& rhs)
{
  if(&rhs!=this)
  {
    mType                 = rhs.mType;
    mChar                 = rhs.mChar;
    mInteger              = rhs.mInteger;
    mReal                 = rhs.mReal;
    mDenominator          = rhs.mDenominator;
    mExponent             = rhs.mExponent;
    hasSemantics          = rhs.hasSemantics;
    mParentSBMLObject     = rhs.mParentSBMLObject;
    mUnits                = rhs.mUnits;
    mId                   = rhs.mId;
    mClass                = rhs.mClass;
    mStyle                = rhs.mStyle;
    mIsBvar               = rhs.mIsBvar;
    mUserData             = rhs.mUserData;
    if (rhs.mName)
    {
      mName = safe_strdup(rhs.mName);
    }
    else
    {
      mName = NULL;
    }

    unsigned int size = mChildren->getSize();
    while (size--) delete static_cast<ASTNode*>( mChildren->remove(0) );
    delete mChildren;
    mChildren = new List();

    for (unsigned int c = 0; c < rhs.getNumChildren(); ++c)
    {
      addChild( rhs.getChild(c)->deepCopy() );
    }

    size = mSemanticsAnnotations->getSize();
    while (size--)  delete static_cast<XMLNode*>(mSemanticsAnnotations->remove(0) );
    delete mSemanticsAnnotations;
    mSemanticsAnnotations = new List();

    for (unsigned int c = 0; c < rhs.getNumSemanticsAnnotations(); ++c)
    {
      addSemanticsAnnotation( rhs.getSemanticsAnnotation(c)->clone() );
    }
    
    delete mDefinitionURL;
    mDefinitionURL        = rhs.mDefinitionURL->clone();	
  }
  return *this;
}

/*
 * Destroys this ASTNode including any child nodes.
 */
LIBSBML_EXTERN
ASTNode::~ASTNode ()
{
  unsigned int size = getNumChildren();


  while (size--) delete static_cast<ASTNode*>( mChildren->remove(0) );
  delete mChildren;

  size = mSemanticsAnnotations->getSize();
  while (size--)  delete static_cast<XMLNode*>(mSemanticsAnnotations->remove(0) );
  delete mSemanticsAnnotations;

  delete mDefinitionURL;
  
  freeName();
}


/*
 * Frees the name of this ASTNode and sets it to NULL.
 * 
 * This operation is only applicable to ASTNodes corresponding to
 * operators, numbers, or AST_UNKNOWN.  This method will have no
 * effect on other types of nodes.
 */
int
ASTNode::freeName ()
{
  if (mName != NULL)
  {
    safe_free(mName);
    mName = NULL;
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_UNEXPECTED_ATTRIBUTE;
  }
}


/*
 * Attempts to convert this ASTNode to a canonical form and returns true if
 * the conversion succeeded, false otherwise.
 *
 * The rules determining the canonical form conversion are as follows:
 *
 *   1. If the node type is AST_NAME and the node name matches
 *   "ExponentialE", "Pi", "True" or "False" the node type is converted to
 *   the corresponding AST_CONSTANT type.
 *
 *   2. If the node type is an AST_FUNCTION and the node name matches an L1
 *   or L2 (MathML) function name, logical operator name, or relational
 *   operator name, the node is converted to the correspnding AST_FUNCTION,
 *   AST_LOGICAL or AST_CONSTANT type.
 *
 * L1 function names are searched first, so canonicalizing "log" will
 * result in a node type of AST_FUNCTION_LN (see L1 Specification,
 * Appendix C).
 *
 * Some canonicalizations result in a structural converion of the nodes (by
 * adding a child).  For example, a node with L1 function name "sqr" and a
 * single child node (the argument) will be transformed to a node of type
 * AST_FUNCTION_POWER with two children.  The first child will remain
 * unchanged, but the second child will be an ASTNode of type AST_INTEGER
 * and a value of 2.  The function names that result in structural changes
 * are: log10, sqr and sqrt.
 */
LIBSBML_EXTERN
bool
ASTNode::canonicalize ()
{
  bool found = false;


  if (mType == AST_NAME)
  {
    found = canonicalizeConstant();
  }

  if (!found && mType == AST_FUNCTION)
  {
    found = canonicalizeFunction();

    if (!found)
    {
      found = canonicalizeLogical();
    }

    if (!found)
    {
      found = canonicalizeRelational();
    }
  }

  return found;
}

/** @cond doxygen-libsbml-internal */
/*
 * Internal helper function for canonicalize().
 */
bool
ASTNode::canonicalizeConstant ()
{
  const int first = AST_CONSTANT_E;
  const int last  = AST_CONSTANT_TRUE;
  const int size  = last - first + 1;

  int  index;
  bool found;


  index = util_bsearchStringsI(AST_CONSTANT_STRINGS, mName, 0, size - 1);
  found = (index < size);

  if (found)
  {
    setType( static_cast<ASTNodeType_t>(first + index) );
  }

  return found;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Internal helper function for canonicalize().
 */
bool
ASTNode::canonicalizeFunction ()
{
  const int first = AST_FUNCTION_ABS;
  const int last  = AST_FUNCTION_TANH;
  const int size  = last - first + 1;

  int  index;
  bool found;


  /**
   * Search for SBML Level 1 function names first.
   */
  found = canonicalizeFunctionL1();

  /**
   * Now Lambda...
   */
  if (!found)
  {
    if ( (found = !strcmp_insensitive(mName, AST_LAMBDA_STRING)) )
    {
      setType(AST_LAMBDA);
    }
  }

  /**
   * ... and finally the L2 (MathML) function names.
   */
  if (!found)
  {
    index = util_bsearchStringsI(AST_FUNCTION_STRINGS, mName, 0, size - 1);
    found = (index < size);

    if (found)
    {
      setType( static_cast<ASTNodeType_t>(first + index) );
    }
  }

  return found;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Internal helper function for canonicalize().
 */
bool
ASTNode::canonicalizeFunctionL1 ()
{
  ASTNode* child;


  if ( !strcmp_insensitive(mName, "acos") )
  {
    setType(AST_FUNCTION_ARCCOS);
  }
  else if ( !strcmp_insensitive(mName, "asin") )
  {
    setType(AST_FUNCTION_ARCSIN);
  }
  else if ( !strcmp_insensitive(mName, "atan") )
  {
    setType(AST_FUNCTION_ARCTAN);
  }
  else if ( !strcmp_insensitive(mName, "ceil") )
  {
    setType(AST_FUNCTION_CEILING);
  }

  /**
   * "log(x)" in L1 is represented as "ln(x)" in L2.
   *
   * Notice, however, that the conversion is performed only if the number of
   * arguments is 1.  Thus "log(5, x)" will still be "log(5, x) when passed
   * through this filter.
   */
  else if ( !strcmp_insensitive(mName, "log") && (getNumChildren() == 1) )
  {
    setType(AST_FUNCTION_LN);
  }

  /**
   * "log10(x)" in L1 is represented as "log(10, x)" in L2.
   */
  else if ( !strcmp_insensitive(mName, "log10") && (getNumChildren() == 1) )
  {
    setType(AST_FUNCTION_LOG);

    child = new ASTNode;
    child->setValue(10);

    prependChild(child);
  }

  /**
   * Here we set the type to AST_FUNCTION_POWER.  We could set it to
   * AST_POWER, but then we would loose the idea that it was a function
   * before it was canonicalized.
   */
  else if ( !strcmp_insensitive(mName, "pow") )
  {
    setType(AST_FUNCTION_POWER);
  }

  /**
   * "sqr(x)" in L1 is represented as "power(x, 2)" in L2.
   */
  else if ( !strcmp_insensitive(mName, "sqr") && (getNumChildren() == 1) )
  {
    setType(AST_FUNCTION_POWER);

    child = new ASTNode;
    child->setValue(2);

    addChild(child);
  }

  /**
   * "sqrt(x) in L1 is represented as "root(2, x)" in L1.
   */
  else if ( !strcmp_insensitive(mName, "sqrt") && (getNumChildren() == 1) )
  {
    setType(AST_FUNCTION_ROOT);

    child = new ASTNode;
    child->setValue(2);

    prependChild(child);
  }

  /**
   * Was a conversion performed?
   */
  return (mType != AST_FUNCTION);
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Internal helper function for canonicalize().
 */
bool
ASTNode::canonicalizeLogical ()
{
  const int first = AST_LOGICAL_AND;
  const int last  = AST_LOGICAL_XOR;
  const int size  = last - first + 1;

  int  index;
  bool found;


  index = util_bsearchStringsI(AST_LOGICAL_STRINGS, mName, 0, size - 1);
  found = (index < size);

  if (found)
  {
    setType( static_cast<ASTNodeType_t>(first + index) );
  }

  return found;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/*
 * Internal helper function for canonicalize().
 */
bool
ASTNode::canonicalizeRelational ()
{
  const int first = AST_RELATIONAL_EQ;
  const int last  = AST_RELATIONAL_NEQ;
  const int size  = last - first + 1;

  int  index;
  bool found;


  index = util_bsearchStringsI(AST_RELATIONAL_STRINGS, mName, 0, size - 1);
  found = (index < size);

  if (found)
  {
    setType( static_cast<ASTNodeType_t>(first + index) );
  }

  return found;
}
/** @endcond */


/*
 * Adds the given node as a child of this ASTNode.  Child nodes are added
 * in-order from "left-to-right".
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 */
LIBSBML_EXTERN
int
ASTNode::addChild (ASTNode* child)
{
  unsigned int numBefore = getNumChildren();
  mChildren->add(child);

  if (getNumChildren() == numBefore + 1)
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}

/*
 * Adds the given node as a child of this ASTNode.  This method adds child
 * nodes from "right-to-left".
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
ASTNode::prependChild (ASTNode* child)
{
  if (child == NULL) return LIBSBML_INVALID_OBJECT;

  unsigned int numBefore = getNumChildren();
  mChildren->prepend(child);

  if (getNumChildren() == numBefore + 1)
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


LIBSBML_EXTERN
int  
ASTNode::removeChild(unsigned int n)
{
  int removed = LIBSBML_INDEX_EXCEEDS_SIZE;
  unsigned int size = getNumChildren();
  if (n < size)
  {
    mChildren->remove(n);
    if (getNumChildren() == size-1)
    {
      removed = LIBSBML_OPERATION_SUCCESS;
    }
  }

  return removed;
}

LIBSBML_EXTERN
int 
ASTNode::replaceChild(unsigned int n, ASTNode *newChild)
{
  if (newChild == NULL) return LIBSBML_INVALID_OBJECT;

  int replaced = LIBSBML_INDEX_EXCEEDS_SIZE;

  unsigned int size = getNumChildren();
  if (n < size)
  {
    mChildren->remove(n);
    if (insertChild(n, newChild) == LIBSBML_OPERATION_SUCCESS)
      replaced = LIBSBML_OPERATION_SUCCESS;    
  }
    
  return replaced;
}

LIBSBML_EXTERN
int 
ASTNode::insertChild(unsigned int n, ASTNode *newChild)
{
  if (newChild == NULL) return LIBSBML_INVALID_OBJECT;

  int inserted = LIBSBML_INDEX_EXCEEDS_SIZE;

  unsigned int i, size = getNumChildren();
  if (n == 0)
  {
    prependChild(newChild);
    inserted = LIBSBML_OPERATION_SUCCESS;
  }
  else if (n <= size) 
  {
    /* starting at the end take each child in the list and prepend it
    * then remove it from the end
    * at the insertion point prepend the newChild
    * eg list: a, b, c 
    * inserting d at position 2
    * list goes: c, a, b :  d, c, a, b : b, d, c, a : a, b, d, c
    */
    for (i = size-1; i >= n; i--)
    {
      prependChild(getChild(size-1));
      mChildren->remove(size);
    }

    prependChild(newChild);

    for (i = 0; i < n; i++)
    {
      prependChild(getChild(size));
      mChildren->remove(size+1);
    }

    if (getNumChildren() == size + 1)
      inserted = LIBSBML_OPERATION_SUCCESS;
  }

  return inserted;
}

/*
 * @return a copy of this ASTNode and all its children.  The caller owns
 * the returned ASTNode and is reponsible for deleting it.
 */
LIBSBML_EXTERN
ASTNode*
ASTNode::deepCopy () const
{
  return new ASTNode(*this);
}


/*
 * @return the nth child of this ASTNode or NULL if this node has no nth
 * child (n > getNumChildren() - 1).
 */
LIBSBML_EXTERN
ASTNode*
ASTNode::getChild (unsigned int n) const
{
  return static_cast<ASTNode*>( mChildren->get(n) );
}


/*
 * @return the left child of this ASTNode.  This is equivalent to
 * getChild(0);
 */
LIBSBML_EXTERN
ASTNode*
ASTNode::getLeftChild () const
{
  return static_cast<ASTNode*>( mChildren->get(0) );
}


/*
 * @return the right child of this ASTNode or NULL if this node has no
 * right child.  If getNumChildren() > 1, then this is equivalent to:
 *
 *   getChild( getNumChildren() - 1);
 */
LIBSBML_EXTERN
ASTNode*
ASTNode::getRightChild () const
{
  unsigned int nc = getNumChildren();


  return (nc > 1) ? static_cast<ASTNode*>( mChildren->get(nc - 1) ): NULL;
}


/*
 * @return the number of children of this ASTNode or 0 is this node has no
 * children.
 */
LIBSBML_EXTERN
unsigned int
ASTNode::getNumChildren () const
{
  return mChildren->getSize();
}


/*
 * Adds the given xmlnode as an annotation of this ASTNode.  
 */
LIBSBML_EXTERN
int 
ASTNode::addSemanticsAnnotation (XMLNode* sAnnotation)
{
  if (sAnnotation == NULL)
  {
    return LIBSBML_OPERATION_FAILED;
  }
  mSemanticsAnnotations->add(sAnnotation);
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * @return the number of annotations of this ASTNode.  
 */
LIBSBML_EXTERN
unsigned int 
ASTNode::getNumSemanticsAnnotations () const
{
  return mSemanticsAnnotations->getSize();
}


/*
 * @return the nth annotation of this ASTNode or NULL if this node has no nth
 * annotation.
 */
LIBSBML_EXTERN
XMLNode* 
ASTNode::getSemanticsAnnotation (unsigned int n) const
{
  return static_cast<XMLNode*>( mSemanticsAnnotations->get(n) );
}

/*
 * Performs a depth-first search (DFS) of the tree rooted at node and
 * returns the List of nodes where predicate(node) returns true.
 *
 * The typedef for ASTNodePredicate is:
 *
 *   int (*ASTNodePredicate) (const ASTNode_t *node);
 *
 * where a return value of non-zero represents true and zero represents
 * false.
 *
 * The List returned is owned by the caller and should be deleted.  The
 * ASTNodes in the list, however, are not owned by the caller (as they
 * still belong to the tree itself) and therefore should not be deleted.
 */
LIBSBML_EXTERN
List*
ASTNode::getListOfNodes (ASTNodePredicate predicate) const
{
  if (predicate == NULL) return NULL;

  List* lst = new List;


  fillListOfNodes(predicate, lst);

  return lst;
}


/*
 * This method is identical in functionality to getListOfNodes(), except
 * the List is passed-in by the caller.
 */
LIBSBML_EXTERN
void
ASTNode::fillListOfNodes (ASTNodePredicate predicate, List* lst) const
{
  if (lst == NULL || predicate == NULL) return;

  ASTNode*     child;
  unsigned int c;
  unsigned int numChildren = getNumChildren();



  if (predicate(this) != 0)
  {
    lst->add( const_cast<ASTNode*>(this) );
  }

  for (c = 0; c < numChildren; c++)
  {
    child = getChild(c);
    child->fillListOfNodes(predicate, lst);
  }
}


/*
 * @return the value of this ASTNode as a single character.  This function
 * should be called only when getType() is one of AST_PLUS, AST_MINUS,
 * AST_TIMES, AST_DIVIDE or AST_POWER.
 */
LIBSBML_EXTERN
char
ASTNode::getCharacter () const
{
  return mChar;
}


/*
 * @return the value of this ASTNode as a (long) integer.  This function
 * should be called only when getType() == AST_INTEGER.
 */
LIBSBML_EXTERN
long
ASTNode::getInteger () const
{
  return mInteger;
}


/*
 * @return the value of this ASTNode as a string.  This function may be
 * called on nodes that are not operators (isOperator() == false)
 * or numbers (isNumber() == false).
 */
LIBSBML_EXTERN
const char*
ASTNode::getName () const
{
  const char* result = mName;


  /**
   * If the node does not have a name and is not a user-defined function
   * (type == AST_FUNCTION), use the default name for the builtin node
   * types.
   */
  if (mName == NULL && mType != AST_FUNCTION)
  {
    if ( isConstant() )
    {
      result = AST_CONSTANT_STRINGS[ mType - AST_CONSTANT_E ];
    }
    else if ( isLambda() )
    {
      result = AST_LAMBDA_STRING;
    }
    else if ( isFunction() )
    {
      result = AST_FUNCTION_STRINGS[ mType - AST_FUNCTION_ABS ];
    }
    else if ( isLogical() )
    {
      result = AST_LOGICAL_STRINGS[ mType - AST_LOGICAL_AND ];
    }
    else if ( isRelational() )
    {
      result = AST_RELATIONAL_STRINGS[ mType - AST_RELATIONAL_EQ ];
    }
  }

  return result;
}


LIBSBML_EXTERN
const char*
ASTNode::getOperatorName () const
{
  switch(mType) {
  case AST_DIVIDE:
    return AST_OPERATOR_STRINGS[0];
  case AST_MINUS:
    return AST_OPERATOR_STRINGS[1];
  case AST_PLUS:
    return AST_OPERATOR_STRINGS[2];
  case AST_TIMES:
    return AST_OPERATOR_STRINGS[3];
  case AST_POWER:
    return AST_OPERATOR_STRINGS[4];
  default:
    return NULL;
  }
}


/*
 * @return the value of the numerator of this ASTNode.  This function
 * should be called only when getType() == AST_RATIONAL.
 */
LIBSBML_EXTERN
long
ASTNode::getNumerator () const
{
  return mInteger;
}


/*
 * @return the value of the denominator of this ASTNode.  This function
 * should be called only when getType() == AST_RATIONAL.
 */
LIBSBML_EXTERN
long
ASTNode::getDenominator () const
{
  return mDenominator;
}


/*
 * @return the value of this ASTNode as a real (double).  This function
 * should be called only when isReal() == true.
 *
 * This function performs the necessary arithmetic if the node type is
 * AST_REAL_E (mantissa * $10^exponent$) or AST_RATIONAL (numerator /
 * denominator).
 */
LIBSBML_EXTERN
double
ASTNode::getReal () const
{
  double result = mReal;
  

  if (mType == AST_REAL_E)
  {
    result *= pow(10.0,  static_cast<double>(mExponent) );
  }
  else if (mType == AST_RATIONAL)
  {
    result = static_cast<double>(mInteger) / mDenominator;
  }

  return result;
}


/*
 * @return the value of the mantissa of this ASTNode.  This function should
 * be called only when getType() is AST_REAL_E or AST_REAL.  If AST_REAL,
 * this method is identical to getReal().
 */
LIBSBML_EXTERN
double
ASTNode::getMantissa () const
{
  return mReal;
}


/*
 * @return the value of the exponent of this ASTNode.  This function should
 * be called only when getType() is AST_REAL_E or AST_REAL.
 */
LIBSBML_EXTERN
long
ASTNode::getExponent () const
{
  return mExponent;
}


/*
 * @return the precedence of this ASTNode (as defined in the SBML L1
 * specification).
 */
LIBSBML_EXTERN
int
ASTNode::getPrecedence () const
{
  int precedence;


  if ( isUMinus() )
  {
    precedence = 5;
  }
  else
  {
    switch (mType)
    {
      case AST_PLUS:
      case AST_MINUS:
        precedence = 2;
        break;

      case AST_DIVIDE:
      case AST_TIMES:
        precedence = 3;
        break;

      case AST_POWER:
        precedence = 4;
        break;

      default:
        precedence = 6;
        break;
    }
  }

  return precedence;
}


/*
 * @return the type of this ASTNode.
 */
LIBSBML_EXTERN
ASTNodeType_t
ASTNode::getType () const
{
  return mType;
}

LIBSBML_EXTERN
std::string
ASTNode::getId() const
{
  return mId;
}

LIBSBML_EXTERN
std::string
ASTNode::getClass() const
{
  return mClass;
}

LIBSBML_EXTERN
std::string
ASTNode::getStyle() const
{
  return mStyle;
}

LIBSBML_EXTERN
std::string
ASTNode::getUnits() const
{
  return mUnits;
}

/*
 * @return true if this ASTNode is a boolean (a logical operator, a
 * relational operator, or the constants true or false), false otherwise.
 */
LIBSBML_EXTERN
bool
ASTNode::isAvogadro () const
{
  return ASTNodeType_isAvogadro(mType);
}


/*
 * @return true if this ASTNode is a boolean (a logical operator, a
 * relational operator, or the constants true or false), false otherwise.
 */
LIBSBML_EXTERN
bool
ASTNode::isBoolean () const
{
  return
    isLogical   () ||
    isRelational() ||
    mType == AST_CONSTANT_TRUE ||
    mType == AST_CONSTANT_FALSE;
}


LIBSBML_EXTERN
bool
ASTNode::returnsBoolean (const Model* givenModel /*=NULL*/) const
{   

  if (isBoolean() == true)
  {
    return true;
  }

  const Model* model = givenModel;
  if (givenModel == NULL && getParentSBMLObject() != NULL)
  {
    model = getParentSBMLObject()->getModel();
  }

  if (getType() == AST_FUNCTION)
  {
    if (model == NULL)
    {
      return false;
    }
    else
    {
      const FunctionDefinition* fd = model->getFunctionDefinition( getName() );

      if (fd != NULL && fd->isSetMath())
      {
        return fd->getMath()->getRightChild()->returnsBoolean();
      }
      else
      {
        return false;
      }
    }
  }

  else if (getType() == AST_FUNCTION_PIECEWISE)
  {
    for (unsigned int c = 0; c < getNumChildren(); c += 2)
    {
      if ( getChild(c)->returnsBoolean() == false ) 
        return false;
    }

    return true;
  }

  // add explicit return value in case we overlooked something
  return false;
}




/*
 * @return true if this ASTNode is a MathML constant (true,
 * false, pi, exponentiale), false otherwise.
 */
LIBSBML_EXTERN
bool
ASTNode::isConstant () const
{
  return ASTNodeType_isConstant(mType);
}


/*
 * @return true if this ASTNode is a function in SBML L1, L2 (MathML)
 * (everything from abs() to tanh()) or user-defined, false otherwise.
 */
LIBSBML_EXTERN
bool
ASTNode::isFunction () const
{
  return ASTNodeType_isFunction(mType);
}


/*
 * @return true if this ASTNode is the special IEEE 754 value infinity,
 * false otherwise.
 */
LIBSBML_EXTERN
bool
ASTNode::isInfinity () const
{
  return isReal() ? util_isInf( getReal() ) > 0 : false;
}


/*
 * @return true if this ASTNode is of type AST_INTEGER, false otherwise.
 */
LIBSBML_EXTERN
bool
ASTNode::isInteger () const
{
  return ASTNodeType_isInteger(mType);
}


/*
 * @return true if this ASTNode is of type AST_LAMBDA, false otherwise.
 */
LIBSBML_EXTERN
bool
ASTNode::isLambda () const
{
  return ASTNodeType_isLambda(mType);
}

/*
 * @return true if the given ASTNode represents a log10() function, false
 * otherwise.
 *
 * More precisley, the node type is AST_FUNCTION_LOG with two children the
 * first of which is an AST_INTEGER equal to 10.
 */
LIBSBML_EXTERN
bool
ASTNode::isLog10 () const
{
  bool     result = false;
  ASTNode* c;


  if (mType == AST_FUNCTION_LOG)
  {
    if (getNumChildren() == 2)
    {
      c = getLeftChild();

      if ((c->mType == AST_INTEGER) && (c->mInteger == 10))
      {
        result = true;
      }
    }
  }

  return result;
}


/*
 * @return true if this ASTNode is a MathML logical operator (and, or, not,
 * xor), false otherwise.
 */
LIBSBML_EXTERN
bool
ASTNode::isLogical () const
{
  return ASTNodeType_isLogical(mType);
}


/*
 * @return true if this ASTNode is a user-defined variable name in SBML L1,
 * L2 (MathML) or the special symbols delay or time, false otherwise.
 */
LIBSBML_EXTERN
bool
ASTNode::isName () const
{
  return ASTNodeType_isName(mType);
}


/*
 * @return true if this ASTNode is the special IEEE 754 value not a
 * number, false otherwise.
 */
LIBSBML_EXTERN
bool
ASTNode::isNaN () const
{
  if ( isReal() )
  {
    double value = getReal();
    return (value != value);
  }

  return false;
}


/*
 * @return true if this ASTNode is the special IEEE 754 value negative
 * infinity, false otherwise.
 */
LIBSBML_EXTERN
bool
ASTNode::isNegInfinity () const
{
  return isReal() ? util_isInf( getReal() ) < 0 : false;
}


/*
 * @return true if this ASTNode is a number, false otherwise.
 *
 * This is functionally equivalent to:
 *
 *   isInteger() || isReal().
 */
LIBSBML_EXTERN
bool
ASTNode::isNumber () const
{
  return ASTNodeType_isNumber(mType);
}


/*
 * @return true if this ASTNode is an operator, false otherwise.  Operators
 * are: +, -, *, / and \^ (power).
 */
LIBSBML_EXTERN
bool
ASTNode::isOperator () const
{
  return ASTNodeType_isOperator(mType);
}


/*
 * @return true if this ASTNode is a piecewise function, false otherwise.
 */
LIBSBML_EXTERN
bool
ASTNode::isPiecewise () const
{
  return (mType == AST_FUNCTION_PIECEWISE);
}


/*
 * @return true if this ASTNode is of type AST_RATIONAL, false otherwise.
 */
LIBSBML_EXTERN
bool
ASTNode::isRational () const
{
  return ASTNodeType_isRational(mType);
}


/*
 * @return true if the value of this ASTNode can represented as a real
 * number, false otherwise.
 *
 * To be a represented as a real number, this node must be of one of the
 * following types: AST_REAL, AST_REAL_E or AST_RATIONAL.
 */
LIBSBML_EXTERN
bool
ASTNode::isReal () const
{
  return ASTNodeType_isReal(mType);
}


/*
 * @return true if this ASTNode is a MathML relational operator (==, >=, >,
 * <=, < !=), false otherwise.
 */
LIBSBML_EXTERN
bool
ASTNode::isRelational () const
{
  return ASTNodeType_isRelational(mType);
}


/*
 * @return true if the given ASTNode represents a sqrt() function, false
 * otherwise.
 *
 * More precisley, the node type is AST_FUNCTION_ROOT with two children the
 * first of which is an AST_INTEGER equal to 2.
 */
LIBSBML_EXTERN
bool
ASTNode::isSqrt () const
{
  int      result = false;
  ASTNode* c;


  if (mType == AST_FUNCTION_ROOT)
  {
    if (getNumChildren() == 2)
    {
      c = getLeftChild();

      if ((c->mType == AST_INTEGER) && (c->mInteger == 2))
      {
        result = true;
      }
    }
  }

  return result;
}


/*
 * @return true if this ASTNode is a unary minus, false otherwise.
 *
 * For numbers, unary minus nodes can be "collapsed" by negating the
 * number.  In fact, SBML_parseFormula() does this during its parse.
 * However, unary minus nodes for symbols (AST_NAMES) cannot be
 * "collapsed", so this predicate function is necessary.
 *
 * A node is defined as a unary minus node if it is of type AST_MINUS and
 * has exactly one child.
 */
LIBSBML_EXTERN
bool
ASTNode::isUMinus () const
{
  bool uminus = false;


  if (mType == AST_MINUS)
  {
    if (getNumChildren() == 1)
    {
      uminus = true;
    }
  }

  return uminus;
}

LIBSBML_EXTERN
bool
ASTNode::isUPlus () const
{
  bool uplus = false;


  if (mType == AST_PLUS)
  {
    if (getNumChildren() == 1)
    {
      uplus = true;
    }
  }

  return uplus;
}

/*
 * @return true if this ASTNode is of type AST_UNKNOWN, false otherwise.
 */
LIBSBML_EXTERN
bool
ASTNode::isUnknown () const
{
  return ASTNodeType_isUnknown(mType);
}


LIBSBML_EXTERN
bool 
ASTNode::isSetId() const
{
  return (mId.empty() == false);
}
  
LIBSBML_EXTERN
bool 
ASTNode::isSetClass() const
{
  return (mClass.empty() == false);
}
  
LIBSBML_EXTERN
bool 
ASTNode::isSetStyle() const
{
  return (mStyle.empty() == false);
}
  
LIBSBML_EXTERN
bool 
ASTNode::isSetUnits() const
{
  return (mUnits.empty() == false);
}
  

LIBSBML_EXTERN
bool 
ASTNode::hasUnits() const
{
  bool hasUnits = isSetUnits();

  unsigned int n = 0;
  while(!hasUnits && n < getNumChildren())
  {
    hasUnits = getChild(n)->hasUnits();
    n++;
  }

  return hasUnits;
}

  
/*
 * Sets the value of this ASTNode to the given character.  If character is
 * one of '+', '-', '*', '/' or '\^', the node type will be set
 * accordingly.  For all other characters, the node type will be set to
 * AST_UNKNOWN.
 */
LIBSBML_EXTERN
int
ASTNode::setCharacter (char value)
{
  setType( static_cast<ASTNodeType_t>(value) );
  mChar = value;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * Sets the value of this ASTNode to the given name.
 *
 * The node type will be set (to AST_NAME) ONLY IF the ASTNode was
 * previously an operator (isOperator(node) == true) or number
 * (isNumber(node) == true).  This allows names to be set for AST_FUNCTIONs
 * and the like.
 */
LIBSBML_EXTERN
int
ASTNode::setName (const char *name)
{
  if (getName() == name) 
    return LIBSBML_OPERATION_SUCCESS;


  if ( isOperator() || isNumber() || isUnknown() )
  {
    mType = AST_NAME;
  }

  freeName();
  mName = (name == NULL) ? NULL : safe_strdup(name);
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * Sets the value of this ASTNode to the given integer and sets the node
 * type to AST_INTEGER.
 */
LIBSBML_EXTERN
int
ASTNode::setValue (int value)
{
  setType(AST_INTEGER);
  mInteger = value;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * Sets the value of this ASTNode to the given (long) integer and sets the
 * node type to AST_INTEGER.
 */
LIBSBML_EXTERN
int
ASTNode::setValue (long value)
{
  setType(AST_INTEGER);
  mInteger = value;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * Sets the value of this ASTNode to the given rational in two parts:
 * the numerator and denominator.  The node type is set to AST_RATIONAL.
 */
LIBSBML_EXTERN
int
ASTNode::setValue (long numerator, long denominator)
{
  setType(AST_RATIONAL);

  mInteger     = numerator;
  mDenominator = denominator;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * Sets the value of this ASTNode to the given real (double) and sets the
 * node type to AST_REAL.
 *
 * This is functionally equivalent to:
 *
 *   setValue(value, 0);
 */
LIBSBML_EXTERN
int
ASTNode::setValue (double value)
{
  setType(AST_REAL);

  mReal     = value;
  mExponent = 0;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * Sets the value of this ASTNode to the given real (double) in two parts:
 * the mantissa and the exponent.  The node type is set to AST_REAL_E.
 */
LIBSBML_EXTERN
int
ASTNode::setValue (double mantissa, long exponent)
{
  setType(AST_REAL_E);

  mReal     = mantissa;
  mExponent = exponent;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * Sets the type of this ASTNode to the given ASTNodeType.
 */
LIBSBML_EXTERN
int
ASTNode::setType (ASTNodeType_t type)
{
  if (mType == type) 
  {
    return LIBSBML_OPERATION_SUCCESS;
  }

  if (isOperator() || isNumber())
  {
    mReal     = 0;
    mExponent = 0;
    mDenominator = 1;
    mInteger = 0;
  }

  /* if avogadro set value */
  if (type == AST_NAME_AVOGADRO)
  {
    /* this will free the name - which people may not want */
    /* this solution does not in fact work for all language bindings */

    //if (mName != NULL)
    //{
    //  char * name = mName;
    //  setValue(6.02214179e23);
    //  setName(name);
    //}
    //else
    //{
      mReal = 6.02214179e23;
    //}
  }

  /**
   * Free name only if the ASTNodeType is being set to something that
   * cannot contain a string.
   *
   * Note: freeName() will only free value.name if there is something to be
   * freed.
   */
  if ( ASTNodeType_isOperator(type) || ASTNodeType_isNumber(type) )
  {
    freeName();
  }

  if ( ASTNodeType_isOperator(type) )
  {
    mType = type;
    mChar = (char) type;
    return LIBSBML_OPERATION_SUCCESS;
  }
  else if ((type >= AST_INTEGER) && (type < AST_UNKNOWN))
  {
    mType = type;
    mChar = 0;
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    mType = AST_UNKNOWN;
    mChar = 0;
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;
  }
}

LIBSBML_EXTERN
int
ASTNode::setId (std::string id)
{
  mId     = id;
  return LIBSBML_OPERATION_SUCCESS;
}

LIBSBML_EXTERN
int
ASTNode::setClass (std::string className)
{
  mClass   = className;
  return LIBSBML_OPERATION_SUCCESS;
}

LIBSBML_EXTERN
int
ASTNode::setStyle (std::string style)
{
  mStyle     = style;
  return LIBSBML_OPERATION_SUCCESS;
}



LIBSBML_EXTERN
int
ASTNode::setUnits (std::string units)
{
  if (!isNumber())
    return LIBSBML_UNEXPECTED_ATTRIBUTE;

  if (!SyntaxChecker::isValidInternalUnitSId(units))
    return LIBSBML_INVALID_ATTRIBUTE_VALUE;

  mUnits     = units;
  return LIBSBML_OPERATION_SUCCESS;
}


/*
 * Swap the children of this ASTNode with the children of that ASTNode.
 */
LIBSBML_EXTERN
int
ASTNode::swapChildren (ASTNode *that)
{
  if (that == NULL)
    return LIBSBML_OPERATION_FAILED;

  List *temp      = this->mChildren;
  this->mChildren = that->mChildren;
  that->mChildren = temp;
  return LIBSBML_OPERATION_SUCCESS;
}

LIBSBML_EXTERN
void 
ASTNode::renameSIdRefs(const std::string& oldid, const std::string& newid)
{
  if (getType() == AST_NAME ||
      getType() == AST_FUNCTION ||
      getType() == AST_UNKNOWN) {
    if (getName() == oldid) {
      setName(newid.c_str());
    }
  }
  for (unsigned int child=0; child<getNumChildren(); child++) {
    getChild(child)->renameSIdRefs(oldid, newid);
  }
}

LIBSBML_EXTERN
void 
ASTNode::renameUnitSIdRefs(const std::string& oldid, const std::string& newid)
{
  if (isSetUnits()) {
    if (getUnits() == oldid) {
      setUnits(newid);
    }
  }
  for (unsigned int child=0; child<getNumChildren(); child++) {
    getChild(child)->renameUnitSIdRefs(oldid, newid);
  }
}


/** @cond doxygen-libsbml-internal */
LIBSBML_EXTERN
void 
ASTNode::replaceIDWithFunction(const std::string& id, const ASTNode* function)
{
  for (unsigned int i=0; i<getNumChildren(); i++) {
    ASTNode* child = getChild(i);
    if (child->getType() == AST_NAME &&
        child->getName() == id) {
      replaceChild(i, function->deepCopy());
    }
    else {
      child->replaceIDWithFunction(id, function);
    }
  }
}
/** @endcond doxygen-libsbml-internal */


/** @cond doxygen-libsbml-internal */
LIBSBML_EXTERN
void ASTNode::multiplyTimeBy(const ASTNode* function)
{
  for (unsigned int i=0; i<getNumChildren(); i++) {
    getChild(i)->multiplyTimeBy(function);
  }
  if (getType() == AST_NAME_TIME) {
    setType(AST_TIMES);
    addChild(function->deepCopy());
    ASTNode* time = new ASTNode(AST_NAME_TIME);
    addChild(time);
  }
}
/** @endcond doxygen-libsbml-internal */

LIBSBML_EXTERN
int
ASTNode::unsetId ()
{
  mId.erase();

  if (mId.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


LIBSBML_EXTERN
int
ASTNode::unsetClass ()
{
  mClass.erase();

  if (mClass.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}

LIBSBML_EXTERN
int
ASTNode::unsetStyle ()
{
  mStyle.erase();

  if (mStyle.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}

LIBSBML_EXTERN
int
ASTNode::unsetUnits ()
{
  if (!isNumber())
    return LIBSBML_UNEXPECTED_ATTRIBUTE;

  mUnits.erase();

  if (mUnits.empty())
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}


/** @cond doxygen-libsbml-internal */
/**
  * Sets the flag indicating that this ASTNode has semantics attached
  */
int 
ASTNode::setSemanticsFlag() 
{ 
  hasSemantics = true; 
  return LIBSBML_OPERATION_SUCCESS;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/**
  * Unsets the flag indicating that this ASTNode has semantics attached
  */
int 
ASTNode::unsetSemanticsFlag()
{ 
  hasSemantics = false; 

  if (hasSemantics)
  {
    return LIBSBML_OPERATION_FAILED;
  }
  else
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/**
  * gets the flag indicating that this ASTNode has semantics attached
  */
bool 
ASTNode::getSemanticsFlag() const
{
  return hasSemantics;
}
/** @endcond */


/** @cond doxygen-libsbml-internal */
/**
  * sets the definitionURL attributes
  */
int 
ASTNode::setDefinitionURL(XMLAttributes url)
{
  delete mDefinitionURL;
  mDefinitionURL = static_cast<XMLAttributes *>(url.clone());
  return LIBSBML_OPERATION_SUCCESS;
}

bool
ASTNode::hasCorrectNumberArguments() const
{
  bool correctNum = true;
  ASTNodeType_t type = getType();
  unsigned int numChildren = getNumChildren();

  switch (type) 
  {
  case AST_INTEGER:
  case AST_REAL:
  case AST_REAL_E:
  case AST_RATIONAL:
  case AST_NAME:
  case AST_NAME_AVOGADRO:
  case AST_NAME_TIME:
  case AST_CONSTANT_E:
  case AST_CONSTANT_FALSE:
  case AST_CONSTANT_PI:
  case AST_CONSTANT_TRUE:

    if (numChildren != 0)
      correctNum = false;
      break;

  case AST_FUNCTION_ABS:
  case AST_FUNCTION_ARCCOS:
  case AST_FUNCTION_ARCCOSH:
  case AST_FUNCTION_ARCCOT:
  case AST_FUNCTION_ARCCOTH:
  case AST_FUNCTION_ARCCSC:
  case AST_FUNCTION_ARCCSCH:
  case AST_FUNCTION_ARCSEC:
  case AST_FUNCTION_ARCSECH:
  case AST_FUNCTION_ARCSIN:
  case AST_FUNCTION_ARCSINH:
  case AST_FUNCTION_ARCTAN:
  case AST_FUNCTION_ARCTANH:
  case AST_FUNCTION_CEILING:
  case AST_FUNCTION_COS:
  case AST_FUNCTION_COSH:
  case AST_FUNCTION_COT:
  case AST_FUNCTION_COTH:
  case AST_FUNCTION_CSC:
  case AST_FUNCTION_CSCH:
  case AST_FUNCTION_EXP:
  case AST_FUNCTION_FACTORIAL:
  case AST_FUNCTION_FLOOR:
  case AST_FUNCTION_LN:
  case AST_FUNCTION_SEC:
  case AST_FUNCTION_SECH:
  case AST_FUNCTION_SIN:
  case AST_FUNCTION_SINH:
  case AST_FUNCTION_TAN:
  case AST_FUNCTION_TANH:
  case AST_LOGICAL_NOT:

    if (numChildren != 1)
      correctNum = false;
      break;

    case AST_DIVIDE:
    case AST_POWER:
    case AST_RELATIONAL_NEQ:
    case AST_FUNCTION_DELAY:
    case AST_FUNCTION_POWER:
    case AST_FUNCTION_LOG:       // a log ASTNode has a child for base

    if (numChildren != 2)
      correctNum = false;
      break;

    case AST_TIMES:
    case AST_PLUS:
    case AST_LOGICAL_AND:
    case AST_LOGICAL_OR:
    case AST_LOGICAL_XOR:
      correctNum = true;
      break;

    case AST_RELATIONAL_EQ:
    case AST_RELATIONAL_GEQ:
    case AST_RELATIONAL_GT:
    case AST_RELATIONAL_LEQ:
    case AST_RELATIONAL_LT:
    case AST_FUNCTION_PIECEWISE:

    if (numChildren < 2)
      correctNum = false;
      break;

    case AST_FUNCTION_ROOT:
    case AST_MINUS:
      
    if (numChildren < 1 || numChildren > 2)
      correctNum = false;
      break;

    case AST_LAMBDA:
      
    if (numChildren < 1)
      correctNum = false;
      break;
     
    case AST_FUNCTION:
      break;

    default:
      break;

  }

  return correctNum;
}
bool 
ASTNode::isWellFormedASTNode() const
{
  bool valid = hasCorrectNumberArguments();
  unsigned int numChildren = getNumChildren();
  unsigned int i = 0;

  // check number of arguments
  while (valid && i < numChildren)
  {
    valid = getChild(i)->isWellFormedASTNode();
    i++;
  }
  return valid;
}


/** @endcond */


LIBSBML_EXTERN
void
ASTNode::replaceArgument(const std::string bvar, ASTNode * arg)
{
  if (arg == NULL)
    return;
  else if (getNumChildren() == 0)
  {
    if (this->isName() && this->getName() == bvar)
    {
      if (arg->isName())
      {
        this->setName(arg->getName());
      }
      else if (arg->isReal())
      {
        this->setValue(arg->getReal());
      }
      else if (arg->isInteger())
      {
        this->setValue(arg->getInteger());
      }
      else if (arg->isConstant())
      {
        this->setType(arg->getType());
      }
      else
      {
        this->setType(arg->getType());
        this->setName(arg->getName());
        for (unsigned int c = 0; c < arg->getNumChildren(); c++)
        {
          this->addChild(arg->getChild(c)->deepCopy());
        }
      }
    }
  }
  for (unsigned int i = 0; i < getNumChildren(); i++)
  {
    if (getChild(i)->isName())
    {
      if (getChild(i)->getName() == bvar)
      {
        if (arg->isName())
        {
          getChild(i)->setName(arg->getName());
        }
        else if (arg->isReal())
        {
          getChild(i)->setValue(arg->getReal());
        }
        else if (arg->isInteger())
        {
          getChild(i)->setValue(arg->getInteger());
        }
        else if (arg->isConstant())
        {
          getChild(i)->setType(arg->getType());
        }
        else
        {
          getChild(i)->setType(arg->getType());
          getChild(i)->setName(arg->getName());
          for (unsigned int c = 0; c < arg->getNumChildren(); c++)
          {
            getChild(i)->addChild(arg->getChild(c)->deepCopy());
          }
        }
      }
    }
    else
    {
      getChild(i)->replaceArgument(bvar, arg);
    }
  }
}

LIBSBML_EXTERN
void
ASTNode::reduceToBinary()
{
  unsigned int numChildren = getNumChildren();
  /* number of children should be greater than 2 */
  if (numChildren < 3)
    return;

  ASTNode* op = new ASTNode( getType() );
  ASTNode* op2 = new ASTNode( getType() );

  // add the first two children to the first node
  op->addChild(getChild(0));
  op->addChild(getChild(1));

  op2->addChild(op);
  for (unsigned int n = 2; n < numChildren; n++)
    op2->addChild(getChild(n));

  swapChildren(op2);

  reduceToBinary();
}


/** @cond doxygen-libsbml-internal */
LIBSBML_EXTERN
void 
ASTNode::setParentSBMLObject(SBase * sb)
{
  mParentSBMLObject = sb;
}
/** @endcond */


LIBSBML_EXTERN
SBase * 
ASTNode::getParentSBMLObject() const
{
  return mParentSBMLObject;
}


/*
  * gets the definitionURL attributes
  */
LIBSBML_EXTERN
XMLAttributes*
ASTNode::getDefinitionURL() const
{
  return mDefinitionURL;
}

LIBSBML_EXTERN
void *
ASTNode::getUserData() const
{
	return this->mUserData;
}


LIBSBML_EXTERN
int
ASTNode::setUserData(void *userData)
{
	this->mUserData = userData;
  if (mUserData != NULL)
  {
    return LIBSBML_OPERATION_SUCCESS;
  }
  else
  {
    return LIBSBML_OPERATION_FAILED;
  }
}

/**
 * Creates a new ASTNode and returns a pointer to it.  The returned node
 * will have a type of AST_UNKNOWN and should be set to something else as
 * soon as possible.
 */
LIBSBML_EXTERN
ASTNode_t *
ASTNode_create (void)
{
  return new(nothrow) ASTNode;
}


/**
 * Creates a new ASTNode and sets its type to the given ASTNodeType.
 */
LIBSBML_EXTERN
ASTNode_t *
ASTNode_createWithType (ASTNodeType_t type)
{
  return new(nothrow) ASTNode(type);
}


/**
 * Creates a new ASTNode from the given Token and returns a pointer to it.
 * The returned ASTNode will contain the same data as the Token.
 */
LIBSBML_EXTERN
ASTNode_t *
ASTNode_createFromToken (Token_t *token)
{
  if (token == NULL) return NULL;
  return new(nothrow) ASTNode(token);
}


/**
 * Frees the given ASTNode including any child nodes.
 */
LIBSBML_EXTERN
void
ASTNode_free (ASTNode_t *node)
{
  if (node == NULL) return;

  delete static_cast<ASTNode*>(node);
}


/**
 * Frees the name of this ASTNode and sets it to NULL, if appropriate,
 * i.e. the node is not an operator, number or AST_UNKNOWN.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t.  @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_UNEXPECTED_ATTRIBUTE
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
ASTNode_freeName (ASTNode_t *node)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return static_cast<ASTNode*>(node)->freeName();
}


/**
 * Attempts to convert this ASTNode to a canonical form and returns true
 * (non-zero) if the conversion succeeded, false (0) otherwise.
 *
 * The rules determining the canonical form conversion are as follows:
 *
 *   1. If the node type is AST_NAME and the node name matches
 *   "ExponentialE", "Pi", "True" or "False" the node type is converted to
 *   the corresponding AST_CONSTANT type.
 *
 *   2. If the node type is an AST_FUNCTION and the node name matches an L1
 *   or L2 (MathML) function name, logical operator name, or relational
 *   operator name, the node is converted to the correspnding AST_FUNCTION,
 *   AST_LOGICAL or AST_CONSTANT type.
 *
 * L1 function names are searched first, so canonicalizing "log" will
 * result in a node type of AST_FUNCTION_LN (see L1 Specification,
 * Appendix C).
 *
 * Some canonicalizations result in a structural converion of the nodes (by
 * adding a child).  For example, a node with L1 function name "sqr" and a
 * single child node (the argument) will be transformed to a node of type
 * AST_FUNCTION_POWER with two children.  The first child will remain
 * unchanged, but the second child will be an ASTNode of type AST_INTEGER
 * and a value of 2.  The function names that result in structural changes
 * are: log10, sqr and sqrt.
 */
LIBSBML_EXTERN
int
ASTNode_canonicalize (ASTNode_t *node)
{
  if (node == NULL) return (int)false;
  return (int) static_cast<ASTNode*>(node)->canonicalize();
}


/**
 * Adds the given node as a child of this ASTNode.  Child nodes are added
 * in-order from "left-to-right".
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t.  @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
ASTNode_addChild (ASTNode_t *node, ASTNode_t *child)
{
	if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return static_cast<ASTNode*>(node)->addChild
                                    ( static_cast<ASTNode*>(child) );
}


/**
 * Adds the given node as a child of this ASTNode.  This method adds child
 * nodes from "right-to-left".
 * 
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
ASTNode_prependChild (ASTNode_t *node, ASTNode_t *child)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return static_cast<ASTNode*>(node)->prependChild
                                    ( static_cast<ASTNode*>(child) );
}


/**
 * @return a copy of this ASTNode and all its children.  The caller owns
 * the returned ASTNode and is reponsible for freeing it.
 */
LIBSBML_EXTERN
ASTNode_t *
ASTNode_deepCopy (const ASTNode_t *node)
{
  if ( node == NULL ) return NULL;
  return
    static_cast<ASTNode_t *>( static_cast<const ASTNode*>(node)->deepCopy() );
}


/**
 * @return the nth child of this ASTNode or NULL if this node has no nth
 * child (n > ASTNode_getNumChildren() - 1).
 */
LIBSBML_EXTERN
ASTNode_t *
ASTNode_getChild (const ASTNode_t *node, unsigned int n)
{
  if (node == NULL) return NULL;
  return static_cast<const ASTNode*>(node)->getChild(n);
}


/**
 * @return the left child of this ASTNode.  This is equivalent to
 * ASTNode_getChild(node, 0);
 */
LIBSBML_EXTERN
ASTNode_t *
ASTNode_getLeftChild (const ASTNode_t *node)
{
  if (node == NULL) return NULL;
  return static_cast<const ASTNode*>(node)->getLeftChild();
}


/**
 * @return the right child of this ASTNode or NULL if this node has no
 * right child.  If ASTNode_getNumChildren(node) > 1, then this is
 * equivalent to:
 *
 *   ASTNode_getChild(node, ASTNode_getNumChildren(node) - 1);
 */
LIBSBML_EXTERN
ASTNode_t *
ASTNode_getRightChild (const ASTNode_t *node)
{
  if (node == NULL) return NULL;
  return static_cast<const ASTNode*>(node)->getRightChild();
}


/**
 * @return the number of children of this ASTNode or 0 is this node has no
 * children.
 */
LIBSBML_EXTERN
unsigned int
ASTNode_getNumChildren (const ASTNode_t *node)
{
  if (node == NULL) return 0;
  return static_cast<const ASTNode*>(node)->getNumChildren();
}


/**
 * Performs a depth-first search (DFS) of the tree rooted at node and
 * returns the List of nodes where predicate(node) returns true.
 *
 * The typedef for ASTNodePredicate is:
 *
 *   int (*ASTNodePredicate) (const ASTNode_t *node);
 *
 * where a return value of non-zero represents true and zero represents
 * false.
 *
 * The List returned is owned by the caller and should be freed with
 * List_free().  The ASTNodes in the list, however, are not owned by the
 * caller (as they still belong to the tree itself) and therefore should
 * not be freed.  That is, do not call List_freeItems().
 */
LIBSBML_EXTERN
List_t *
ASTNode_getListOfNodes (const ASTNode_t *node, ASTNodePredicate predicate)
{
  if (node == NULL) return NULL;
  return static_cast<const ASTNode*>(node)->getListOfNodes(predicate);
}


/**
 * This method is identical in functionality to ASTNode_getListOfNodes(),
 * except the List is passed-in by the caller.
 */
LIBSBML_EXTERN
void
ASTNode_fillListOfNodes ( const ASTNode_t  *node,
                          ASTNodePredicate predicate,
                          List_t           *lst )
{
  if (node == NULL) return;

  List* x = static_cast<List*>(lst);

  static_cast<const ASTNode*>(node)->fillListOfNodes(predicate, x);
}


/**
 * @return the value of this ASTNode as a single character.  This function
 * should be called only when ASTNode_getType() is one of AST_PLUS,
 * AST_MINUS, AST_TIMES, AST_DIVIDE or AST_POWER.
 */
LIBSBML_EXTERN
char
ASTNode_getCharacter (const ASTNode_t *node)
{
  if (node == NULL) return CHAR_MAX;
  return static_cast<const ASTNode*>(node)->getCharacter();
}


/**
 * @return the value of this ASTNode as a (long) integer.  This function
 * should be called only when ASTNode_getType() == AST_INTEGER.
 */
LIBSBML_EXTERN
long
ASTNode_getInteger (const ASTNode_t *node)
{
  if (node == NULL) return LONG_MAX;
  return static_cast<const ASTNode*>(node)->getInteger();
}


/**
 * @return the value of this ASTNode as a string.  This function may be
 * called on nodes that are not operators (ASTNode_isOperator(node) == 0)
 * or numbers (ASTNode_isNumber(node) == 0).
 */
LIBSBML_EXTERN
const char *
ASTNode_getName (const ASTNode_t *node)
{
  if (node == NULL) return NULL;
  return static_cast<const ASTNode*>(node)->getName();
}


/**
 * @return the value of the numerator of this ASTNode.  This function
 * should be called only when ASTNode_getType() == AST_RATIONAL.
 */
LIBSBML_EXTERN
long
ASTNode_getNumerator (const ASTNode_t *node)
{
  if (node == NULL) return LONG_MAX;
  return static_cast<const ASTNode*>(node)->getNumerator();
}


/**
 * @return the value of the denominator of this ASTNode.  This function
 * should be called only when ASTNode_getType() == AST_RATIONAL.
 */
LIBSBML_EXTERN
long
ASTNode_getDenominator (const ASTNode_t *node)
{
  if (node == NULL) return LONG_MAX;
  return static_cast<const ASTNode*>(node)->getDenominator();
}


/**
 * @return the value of this ASTNode as a real (double).  This function
 * should be called only when ASTNode_isReal(node) != 0.
 *
 * This function performs the necessary arithmetic if the node type is
 * AST_REAL_E (mantissa * $10^exponent$) or AST_RATIONAL
 * (numerator / denominator).
 */
LIBSBML_EXTERN
double
ASTNode_getReal (const ASTNode_t *node)
{
  if (node == NULL) return util_NaN();
  return static_cast<const ASTNode*>(node)->getReal();
}


/**
 * @return the value of the mantissa of this ASTNode.  This function should
 * be called only when ASTNode_getType() is AST_REAL_E or AST_REAL.  If
 * AST_REAL, this method is identical to ASTNode_getReal().
 */
LIBSBML_EXTERN
double
ASTNode_getMantissa (const ASTNode_t *node)
{
  if (node == NULL) return numeric_limits<double>::quiet_NaN();
  return static_cast<const ASTNode*>(node)->getMantissa();
}


/**
 * @return the value of the exponent of this ASTNode.  This function should
 * be called only when ASTNode_getType() is AST_REAL_E or AST_REAL.
 */
LIBSBML_EXTERN
long
ASTNode_getExponent (const ASTNode_t *node)
{
  if (node == NULL) return LONG_MAX;
  return static_cast<const ASTNode*>(node)->getExponent();
}


/**
 * @return the precedence of this ASTNode (as defined in the SBML L1
 * specification).
 */
LIBSBML_EXTERN
int
ASTNode_getPrecedence (const ASTNode_t *node)
{
  if (node == NULL) return 6; // default precedence
  return static_cast<const ASTNode*>(node)->getPrecedence();
}


/**
 * @return the type of this ASTNode.
 */
LIBSBML_EXTERN
ASTNodeType_t
ASTNode_getType (const ASTNode_t *node)
{
  if (node == NULL) return AST_UNKNOWN;
  return static_cast<const ASTNode*>(node)->getType();
}

/**
 * Get the mathml id of this ASTNode.  
 * 
 * @return the mathml id of this ASTNode.
 */
LIBSBML_EXTERN
const char *
ASTNode_getId(const ASTNode_t * node)
{
  if (node == NULL)
    return NULL;

  return node->getId().empty() ? "" : safe_strdup(node->getId().c_str());
}

/**
 * Get the mathml class of this ASTNode.  
 * 
 * @return the mathml class of this ASTNode.
 */
LIBSBML_EXTERN
const char *
ASTNode_getClass(const ASTNode_t * node)
{
  if (node == NULL)
    return NULL;

  return node->getClass().empty() ? "" : safe_strdup(node->getClass().c_str());
}

/**
 * Get the mathml style of this ASTNode.  
 * 
 * @return the mathml style of this ASTNode.
 */
LIBSBML_EXTERN
const char *
ASTNode_getStyle(const ASTNode_t * node)
{
  if (node == NULL)
    return NULL;

  return node->getStyle().empty() ? "" : safe_strdup(node->getStyle().c_str());
}



/**
 * Get the units of this ASTNode.  
 * 
 * Only applies to cn elements
 *
 * @return the units of this ASTNode.
 */
LIBSBML_EXTERN
const char *
ASTNode_getUnits(const ASTNode_t * node)
{
  if (node == NULL) return NULL;
  return safe_strdup(node->getUnits().c_str());
}


/**
 * @return true (non-zero) if this ASTNode is the special symbol
 * avogadro, false (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isAvogadro (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return (int) static_cast<const ASTNode*>(node)->isAvogadro();
}


/**
 * @return true (non-zero) if this ASTNode is a boolean (a logical
 * operator, a relational operator, or the constants true or false), false
 * (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isBoolean (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return (int) static_cast<const ASTNode*>(node)->isBoolean();
}


LIBSBML_EXTERN
int
ASTNode_returnsBoolean (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return (int) static_cast<const ASTNode*>(node)->returnsBoolean();
}

/**
 * @return true (non-zero) if this ASTNode returns a boolean, false (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_returnsBooleanForModel (const ASTNode_t *node, const Model_t* model)
{
  if (node == NULL) return (int) false;
  return (int) static_cast<const ASTNode*>(node)->returnsBoolean(model);
}



/**
 * @return true (non-zero) if this ASTNode is a MathML constant (true,
 * false, pi, exponentiale), false (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isConstant (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return (int) static_cast<const ASTNode*>(node)->isConstant();
}


/**
 * @return true (non-zero) if this ASTNode is a function in SBML L1, L2
 * (MathML) (everything from abs() to tanh()) or user-defined, false (0)
 * otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isFunction (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return (int) static_cast<const ASTNode*>(node)->isFunction();
}


/**
 * @return true if this ASTNode is the special IEEE 754 value infinity,
 * false otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isInfinity (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return static_cast<int>( node->isInfinity() );
}


/**
 * @return true (non-zero) if this ASTNode is of type AST_INTEGER, false
 * (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isInteger (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return (int) static_cast<const ASTNode*>(node)->isInteger();
}


/**
 * @return true (non-zero) if this ASTNode is of type AST_LAMBDA, false
 * (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isLambda (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return (int) static_cast<const ASTNode*>(node)->isLambda();
}


/**
 * @return true (non-zero) if the given ASTNode represents a log10()
 * function, false (0) otherwise.
 *
 * More precisley, the node type is AST_FUNCTION_LOG with two children the
 * first of which is an AST_INTEGER equal to 10.
 */
LIBSBML_EXTERN
int
ASTNode_isLog10 (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return (int) static_cast<const ASTNode*>(node)->isLog10();
}


/**
 * @return true (non-zero) if this ASTNode is a MathML logical operator
 * (and, or, not, xor), false (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isLogical (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return (int) static_cast<const ASTNode*>(node)->isLogical();
}


/**
 * @return true (non-zero) if this ASTNode is a user-defined variable name
 * in SBML L1, L2 (MathML) or the special symbols delay or time, false (0)
 * otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isName (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return (int) static_cast<const ASTNode*>(node)->isName();
}


/**
 * @return true (non-zero) if this ASTNode is the special IEEE 754 value
 * not a number, false (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isNaN (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return static_cast<int>( node->isNaN() );
}


/**
 * @return true if this ASTNode is the special IEEE 754 value negative
 * infinity, false otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isNegInfinity (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return static_cast<int>( node->isNegInfinity() );
}


/**
 * @return true (non-zero) if this ASTNode is a number, false (0)
 * otherwise.
 *
 * This is functionally equivalent to:
 *
 *   ASTNode_isInteger(node) || ASTNode_isReal(node).
 */
LIBSBML_EXTERN
int
ASTNode_isNumber (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return (int) static_cast<const ASTNode*>(node)->isNumber();
}


/**
 * @return true (non-zero) if this ASTNode is an operator, false (0)
 * otherwise.  Operators are: +, -, *, / and \^ (power).
 */
LIBSBML_EXTERN
int
ASTNode_isOperator (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return (int) static_cast<const ASTNode*>(node)->isOperator();
}


/**
 * @return true (non-zero) if this ASTNode is a piecewise function, false
 * (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isPiecewise (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return static_cast<int>( node->isPiecewise() );
}


/**
 * @return true (non-zero) if this ASTNode is of type AST_RATIONAL, false
 * (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isRational (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return (int) static_cast<const ASTNode*>(node)->isRational();
}


/**
 * @return true (non-zero) if the value of this ASTNode can represented as
 * a real number, false (0) otherwise.
 *
 * To be a represented as a real number, this node must be of one of the
 * following types: AST_REAL, AST_REAL_E or AST_RATIONAL.
 */
LIBSBML_EXTERN
int
ASTNode_isReal (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return (int) static_cast<const ASTNode*>(node)->isReal();
}


/**
 * @return true (non-zero) if this ASTNode is a MathML relational operator
 * (==, >=, >, <=, < !=), false (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isRelational (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return (int) static_cast<const ASTNode*>(node)->isRelational();
}


/**
 * @return true (non-zero) if the given ASTNode represents a sqrt()
 * function, false (0) otherwise.
 *
 * More precisley, the node type is AST_FUNCTION_ROOT with two children the
 * first of which is an AST_INTEGER equal to 2.
 */
LIBSBML_EXTERN
int
ASTNode_isSqrt (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return (int) static_cast<const ASTNode*>(node)->isSqrt();
}


/**
 * @return true (non-zero) if this ASTNode is a unary minus, false (0)
 * otherwise.
 *
 * For numbers, unary minus nodes can be "collapsed" by negating the
 * number.  In fact, SBML_parseFormula() does this during its parse.
 * However, unary minus nodes for symbols (AST_NAMES) cannot be
 * "collapsed", so this predicate function is necessary.
 *
 * A node is defined as a unary minus node if it is of type AST_MINUS and
 * has exactly one child.
 */
LIBSBML_EXTERN
int
ASTNode_isUMinus (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return (int) static_cast<const ASTNode*>(node)->isUMinus();
}

/**
 * @return true (non-zero) if this ASTNode is a unary plus, false (0)
 * otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isUPlus (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return (int) static_cast<const ASTNode*>(node)->isUPlus();
}




/**
 * @return true (non-zero) if this ASTNode is of type AST_UNKNOWN, false
 * (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isUnknown (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return (int) static_cast<const ASTNode*>(node)->isUnknown();
}


/**
 * @return true (non-zero) if this ASTNode has a mathml id, false
 * (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isSetId (const ASTNode_t *node)
{
  return static_cast<int>(node->isSetId());
}

/**
 * @return true (non-zero) if this ASTNode has a mathml class, false
 * (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isSetClass (const ASTNode_t *node)
{
  return static_cast<int>(node->isSetClass());
}

/**
 * @return true (non-zero) if this ASTNode has a mathml style, false
 * (0) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isSetStyle (const ASTNode_t *node)
{
  return static_cast<int>(node->isSetStyle());
}

/**
 * Predicate returning true (non-zero) if this node has sbml:units.
 * 
 * Only applies to cn elements.
 * 
 * @return true if this ASTNode has units, false otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_isSetUnits (const ASTNode_t *node)
{
  if (node == NULL) return (int) false;
  return static_cast<int>(node->isSetUnits());
}


/**
 * Predicate returning true (non-zero) if this node 
 * or any of its children nodes have sbml:units.
 * 
 * @return true if this ASTNode or its children has units, 
 * false otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_hasUnits (const ASTNode_t *node)
{
  if (node == NULL) return (int)false;
  return static_cast<int>(node->hasUnits());
}


/**
 * Sets the value of this ASTNode to the given character.  If character is
 * one of '+', '-', '*', '/' or '\^', the node type will be set accordingly.
 * For all other characters, the node type will be set to AST_UNKNOWN.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t.  @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
ASTNode_setCharacter (ASTNode_t *node, char value)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return static_cast<ASTNode*>(node)->setCharacter(value);
}


/**
 * Sets the value of this ASTNode to the given name.
 *
 * The node type will be set (to AST_NAME) ONLY IF the ASTNode was
 * previously an operator (ASTNode_isOperator(node) != 0) or number
 * (ASTNode_isNumber(node) != 0).  This allows names to be set for
 * AST_FUNCTIONs and the like.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t.  @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
ASTNode_setName (ASTNode_t *node, const char *name)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return static_cast<ASTNode*>(node)->setName(name);
}


/**
 * Sets the value of this ASTNode to the given (long) integer and sets the
 * node type to AST_INTEGER.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t.  @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
ASTNode_setInteger (ASTNode_t *node, long value)
{
  if(node == NULL) return LIBSBML_INVALID_OBJECT;
  return static_cast<ASTNode*>(node)->setValue(value);
}


/**
 * Sets the value of this ASTNode to the given rational in two parts:
 * the numerator and denominator.  The node type is set to AST_RATIONAL.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t.  @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
ASTNode_setRational (ASTNode_t *node, long numerator, long denominator)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return static_cast<ASTNode*>(node)->setValue(numerator, denominator);
}


/**
 * Sets the value of this ASTNode to the given real (double) and sets the
 * node type to AST_REAL.
 *
 * This is functionally equivalent to:
 *
 *   ASTNode_setRealWithExponent(node, value, 0);
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t.  @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
ASTNode_setReal (ASTNode_t *node, double value)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return static_cast<ASTNode*>(node)->setValue(value);
}


/**
 * Sets the value of this ASTNode to the given real (double) in two parts:
 * the mantissa and the exponent.  The node type is set to AST_REAL_E.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t.  @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
ASTNode_setRealWithExponent (ASTNode_t *node, double mantissa, long exponent)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return static_cast<ASTNode*>(node)->setValue(mantissa, exponent);
}


/**
 * Sets the type of this ASTNode to the given ASTNodeType_t.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t.  @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
ASTNode_setType (ASTNode_t *node, ASTNodeType_t type)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return static_cast<ASTNode*>(node)->setType(type);
}


/**
 * Sets the mathml id of this ASTNode to id.
 *
 * @param node the ASTNode to set
 * @param id @c string representing the identifier.
 *
 * @return integer value indicating success/failure of the
 * function.  The possible values returned by this function are:
 * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
 */
LIBSBML_EXTERN
int
ASTNode_setId (ASTNode_t *node, const char *id)
{
  return static_cast<ASTNode*>(node)->setId(id);
}

/**
 * Sets the mathml class of this ASTNode to className.
 *
 * @param node the ASTNode to set
 * @param className @c string representing the identifier.
 *
 * @return integer value indicating success/failure of the
 * function.  The possible values returned by this function are:
 * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
 */
LIBSBML_EXTERN
int
ASTNode_setClass (ASTNode_t *node, const char *className)
{
  return static_cast<ASTNode*>(node)->setClass(className);
}

/**
 * Sets the mathml style of this ASTNode to style.
 *
 * @param node the ASTNode to set
 * @param style @c string representing the identifier.
 *
 * @return integer value indicating success/failure of the
 * function.  The possible values returned by this function are:
 * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
 */
LIBSBML_EXTERN
int
ASTNode_setStyle (ASTNode_t *node, const char *style)
{
  return static_cast<ASTNode*>(node)->setStyle(style);
}

/**
 * Sets the units of this ASTNode to units.
 *
 * The units will be set <em>only if</em> the
 * ASTNode represents an &lt;cn&gt; element. Use isNumber().
 *
 * @param node the ASTNode to set
 * @param units @c string representing units
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_UNEXPECTED_ATTRIBUTE
 * @li LIBSBML_INVALID_ATTRIBUTE_VALUE
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
ASTNode_setUnits (ASTNode_t *node, const char *units)
{
  if (node == NULL ) return LIBSBML_INVALID_OBJECT;
  return static_cast<ASTNode*>(node)->setUnits(units);
}

/**
 * Swap the children of this ASTNode with the children of that ASTNode.
 *
 * @param node the first ASTNode_t structure
 * @param that ASTNode_t structure with which children will swapped
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t.  @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
ASTNode_swapChildren (ASTNode_t *node, ASTNode_t *that)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return static_cast<ASTNode*>(node)
                         ->swapChildren( static_cast<ASTNode*>(that) );
}


/**
 * Unsets the mathml id of this ASTNode.
 *
 * @return integer value indicating success/failure of the
 * function.  The possible values returned by this function are:
 * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
 * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
 */
LIBSBML_EXTERN
int
ASTNode_unsetId (ASTNode_t *node)
{
  return static_cast<ASTNode*>(node)->unsetId();
}

/**
 * Unsets the mathml class of this ASTNode.
 *
 * @return integer value indicating success/failure of the
 * function.  The possible values returned by this function are:
 * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
 * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
 */
LIBSBML_EXTERN
int
ASTNode_unsetClass (ASTNode_t *node)
{
  return static_cast<ASTNode*>(node)->unsetClass();
}

/**
 * Unsets the mathml style of this ASTNode.
 *
 * @return integer value indicating success/failure of the
 * function.  The possible values returned by this function are:
 * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
 * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
 */
LIBSBML_EXTERN
int
ASTNode_unsetStyle (ASTNode_t *node)
{
  return static_cast<ASTNode*>(node)->unsetStyle();
}

/**
 * Unsets the units of this ASTNode.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t. @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_UNEXPECTED_ATTRIBUTE
 * @li LIBSBML_OPERATION_FAILED
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
ASTNode_unsetUnits (ASTNode_t *node)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return static_cast<ASTNode*>(node)->unsetUnits();
}

/**
  * Replaces occurences of a name within the given ASTNode_t structure
  * with the name/value/formula represented by the second argument ASTNode_t
  * e.g. if the formula in this ASTNode is x + y; bvar is x and arg is an 
  * ASTNode representing the real value 3 replaceArgument substitutes 3 for
  * x within this ASTNode
  *
  * @param node the ASTNode_t structure to replace argument
  * @param bvar a string representing the variable name to be substituted
  * @param arg an ASTNode representing the name/value/formula to substitute
  */
LIBSBML_EXTERN
void
ASTNode_replaceArgument(ASTNode_t* node, const char * bvar, ASTNode_t* arg)
{
  if (node == NULL) return ;
  static_cast<ASTNode*>(node)->replaceArgument(bvar, 
                                                  static_cast<ASTNode*>(arg));
}

/**
  * Reduces the given ASTNode_t structure to a binary tree
  * e.g. if the formula in this ASTNode is and(x, y, z) then the 
  * formula of the reduced node would be and(and(x, y), z)
  *
  * @param node the ASTNode_t structure to reduce
  */
LIBSBML_EXTERN
void
ASTNode_reduceToBinary(ASTNode_t* node)
{
  if (node == NULL) return;
  static_cast<ASTNode*>(node)->reduceToBinary();
}

/**
  * Returns the parent SBML structure of the given ASTNode_t structure.
  *
  * @param node the ASTNode_t of which to return the parent.
  * 
  * @return the parent SBML SBase_t structure of this ASTNode_t.
  */
LIBSBML_EXTERN
SBase_t * 
ASTNode_getParentSBMLObject(ASTNode_t* node)
{
  if (node == NULL) return NULL;
  return node->getParentSBMLObject();
}

/**
 * Removes child n of the given ASTNode_t structure. 
 *
 * @param node the ASTNode_t from which to remove a child.
 * @param n unsigned int the index of the child to remove
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t.  @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INDEX_EXCEEDS_SIZE
 * @li LIBSBML_INVALID_OBJECT
 *
 * @note removing a child from an ASTNode may result in an
 * inaccurate representation.
 */
LIBSBML_EXTERN
int
ASTNode_removeChild(ASTNode_t* node, unsigned int n)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return node->removeChild(n);
}

/**
 * Replaces the nth child of the given ASTNode_t structure 
 * with the second ASTNode_t structure.
 *
 * @param node the ASTNode_t within which to replace a child.
 * @param n unsigned int the index of the child to replace
 * @param newChild ASTNode_t structure to replace the nth child
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t.  @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INDEX_EXCEEDS_SIZE
 *
 * @note replacing a child within an ASTNode may result in an
 * inaccurate representation.
 */
LIBSBML_EXTERN
int
ASTNode_replaceChild(ASTNode_t* node, unsigned int n, ASTNode_t * newChild)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return node->replaceChild(n, newChild);
}

/**
 * Insert the second ASTNode_t structure at point n in the list of children
 * of the first ASTNode_t structure.
 *
 * @param node the ASTNode_t within which to insert a child.
 * @param n unsigned int the index of the ASTNode_t being added
 * @param newChild ASTNode_t to insert as the nth child
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t.  @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_INDEX_EXCEEDS_SIZE
 * @li LIBSBML_INVALID_OBJECT
 *
 * @note inserting a child within an ASTNode may result in an
 * inaccurate representation.
 */
LIBSBML_EXTERN
int
ASTNode_insertChild(ASTNode_t* node, unsigned int n, ASTNode_t * newChild)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return node->insertChild(n, newChild);
}

/**
 * Adds the given XMLNode as a semantic annotation of this ASTNode.
 *
 * @param node the ASTNode_t to which the annotation is to be added.
 * @param annotation the XMLNode_t representing the annotation to add.
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t.  @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
ASTNode_addSemanticsAnnotation(ASTNode_t* node, XMLNode_t * annotation)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return node->addSemanticsAnnotation(annotation);
}

/**
 * Get the number of semantic annotation elements inside this node.
 * 
 * @param node the ASTNode_t structure to query.
 *
 * @return the number of annotations of this ASTNode.  
 */
LIBSBML_EXTERN
unsigned int
ASTNode_getNumSemanticsAnnotations(ASTNode_t* node)
{
  if (node == NULL) return 0;
  return node->getNumSemanticsAnnotations();
}

/**
 * Get the nth semantic annotation of this node.
 * 
 * @param node the ASTNode_t structure to query.
 * @param n unsigned int the index of the annotation to be retrieved.
 * 
 * @return the nth annotation of this ASTNode, or NULL if this node has no nth
 * annotation (<code>n &gt; ASTNode_getNumSemanticsAnnotations() - 1</code>).
 */
LIBSBML_EXTERN
XMLNode_t *
ASTNode_getSemanticsAnnotation(ASTNode_t* node, unsigned int n)
{
  if (node == NULL) return NULL;
  return node->getSemanticsAnnotation(n);
}
/**
 * Sets the user data of this node. This can be used by the application
 * developer to attach custom information to the node. In case of a deep
 * copy this attribute will passed as it is. The attribute will be never
 * interpreted by this class.
 * 
 * @param node defines the node of which the user data should be set.
 * @param userData specifies the new user data. 
 *
 * @return integer value indicating success/failure of the
 * function.  @if clike The value is drawn from the
 * enumeration #OperationReturnValues_t.  @endif@~ The possible values
 * returned by this function are:
 * @li LIBSBML_OPERATION_SUCCESS
 * @li LIBSBML_OPERATION_FAILED
 * @li LIBSBML_INVALID_OBJECT
 */
LIBSBML_EXTERN
int
ASTNode_setUserData(ASTNode_t* node, void *userData)
{
  if (node == NULL) return LIBSBML_INVALID_OBJECT;
  return node->setUserData(userData);
}


/**
 * Returns the user data that has been previously set by setUserData().
 *
 * @param node defines the node of interest.
 * @return the user data of this node. NULL if no user data has been.
 * @see ASTNode_setUserData
 */
LIBSBML_EXTERN
void *ASTNode_getUserData(ASTNode_t* node)
{
  if (node == NULL) return NULL;
  return node->getUserData();
}

/**
 * Predicate returning non-zero (true) or zero (false) depending on whether this
 * ASTNode has the correct number of children for it's type.
 *
 * For example, an ASTNode with type AST_PLUS expects 2 child nodes.
 *
 * @note This function performs a check on the toplevel node only.
 * Child nodes are not checked.
 *
 * @see ASTNode_isWellFormedASTNode()
 *
 * @param node defines the node of interest.
 *
 * @return non-zero (true) if this ASTNode is has appropriate number of children
 * for it's type, zero (false) otherwise.
 */
LIBSBML_EXTERN
int
ASTNode_hasCorrectNumberArguments(ASTNode_t* node)
{
  if (node == NULL) return (int)false;
  return static_cast <int> (node->hasCorrectNumberArguments());
}

/**
 * Predicate returning non-zero (true) or zero (false) depending on whether this
 * ASTNode is well-formed.
 *
 * @note An ASTNode may be well-formed, i.e. each node and it's children
 * have the appropriate number of children for the given type, but may still
 * be invalid in the context of an SBML model.
 *
 * @see ASTNode_hasCorrectNumberArguments()
 *
 * @param node defines the node of interest.
 *
 * @return non-zero (true) if this ASTNode is well-formed, zero (false) otherwise.
*/
LIBSBML_EXTERN
int
ASTNode_isWellFormedASTNode(ASTNode_t* node)
{
  if (node == NULL) return (int) false;
  return static_cast <int> (node->isWellFormedASTNode());
}

/** @cond doxygen-libsbml-internal */
/*
 * Internal utility function used in some language binding code.
 */
LIBSBML_EXTERN
int
ASTNode_true(const ASTNode *node)
{
  return 1;
}
/** @endcond */


LIBSBML_CPP_NAMESPACE_END
