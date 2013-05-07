/**
 * @file    ListOf.h
 * @author  Wraps List and inherits from SBase
 * @author  SBML Team <sbml-team@caltech.edu>
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
 * in the file named "LICENSE.txt" included with this software distribution
 * and also available online as http://sbml.org/software/libsbml/license.html
 * ---------------------------------------------------------------------- -->
 *
 * @class ListOf
 * @brief Parent class for the various SBML "ListOfXYZ" classes.
 *
 * @htmlinclude not-sbml-warning.html
 *
 * SBML defines various ListOf___ classes that are containers used for
 * organizing the main components of an SBML model.  All are derived from
 * the abstract class SBase, and inherit the attributes and subelements of
 * SBase, such as "metaid" as and "annotation".  The ListOf___ classes do
 * not add any attributes of their own.
 *
 * The ListOf class in libSBML is a utility class that serves as the parent
 * class for implementing the ListOf__ classes.  It provides methods for
 * working generically with the various SBML lists of objects in a program.
 * LibSBML uses this separate list class rather than ordinary
 * @if clike C&#43;&#43; @endif@if java Java@endif@if python Python@endif@~ lists,
 * so that it can provide the methods and features associated with SBase.
 *
 * @see ListOfFunctionDefinitions
 * @see ListOfUnitDefinitions
 * @see ListOfCompartmentTypes
 * @see ListOfSpeciesTypes
 * @see ListOfCompartments
 * @see ListOfSpecies
 * @see ListOfParameters
 * @see ListOfInitialAssignments
 * @see ListOfRules
 * @see ListOfConstraints
 * @see ListOfReactions
 * @see ListOfEvents
 */


#ifndef ListOf_h
#define ListOf_h


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>
#include <sbml/SBMLTypeCodes.h>


#ifdef __cplusplus


#include <vector>
#include <algorithm>
#include <functional>

#include <sbml/SBase.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class SBMLVisitor;


/** @cond doxygen-libsbml-internal */
/**
 * Used by ListOf::get() to lookup an SBase based by its id.
 */
#ifndef SWIG
template<class CNAME>
struct IdEq : public std::unary_function<SBase*, bool>
{
  const std::string& id;

  IdEq (const std::string& id) : id(id) { }
  bool operator() (SBase* sb) 
       { return static_cast <CNAME*> (sb)->getId() == id; }
};
#endif /* SWIG */
/** @endcond */


class LIBSBML_EXTERN ListOf : public SBase
{
public:

  /**
   * Creates a new ListOf object.
   *
   * @param level the SBML Level; if not assigned, defaults to the
   * value of SBML_DEFAULT_LEVEL.
   * 
   * @param version the Version within the SBML Level; if not assigned,
   * defaults to the value of SBML_DEFAULT_VERSION.
   * 
   * @if notcpp @htmlinclude warn-default-args-in-docs.html @endif@~
   */
  ListOf (unsigned int level   = SBML_DEFAULT_LEVEL, 
          unsigned int version = SBML_DEFAULT_VERSION);
          

  /**
   * Creates a new ListOf with SBMLNamespaces object.
   *
   * @param sbmlns the set of namespaces that this ListOf should contain.
   */
  ListOf (SBMLNamespaces* sbmlns);


  /**
   * Destroys the given ListOf and the items inside it.
   */
  virtual ~ListOf ();


  /**
   * Copy constructor;  creates a copy of this ListOf.
   *
   * @param orig the ListOf instance to copy.
   */
  ListOf (const ListOf& orig);


  /**
   * Assignment operator for ListOf.
   */
  ListOf& operator=(const ListOf& rhs);


  /**
   * Accepts the given SBMLVisitor.
   *
   * @param v the SBMLVisitor instance to be used.
   * 
   * @return the result of calling <code>v.visit()</code>, which indicates
   * whether the Visitor would like to visit the next item in the
   * list.
   */
  virtual bool accept (SBMLVisitor& v) const;


  /**
   * Creates and returns a deep copy of this ListOf.
   * 
   * @return a (deep) copy of this ListOf.
   */
  virtual ListOf* clone () const;


  /**
   * Adds item to the end of this ListOf.
   *
   * This variant of the method makes a clone of the @p item handed to it.
   * This means that when the ListOf is destroyed, the original items will
   * not be destroyed.
   *
   * @param item the item to be added to the list.
   *
   * @see appendAndOwn(SBase* item)
   */
  int append (const SBase* item);


  /**
   * Adds item to the end of this ListOf.
   *
   * This variant of the method does not clone the @p item handed to it;
   * instead, it assumes ownership of it.  This means that when the ListOf
   * is destroyed, the item will be destroyed along with it.
   *
   * @param item the item to be added to the list.
   *
   * @see append(const SBase* item)
   */
  int appendAndOwn (SBase* item);


  /**
   * Adds a clone of all items in the provided ListOf to this object.  This means that when this ListOf is destroyed, the original items will not be destroyed.
   *
   * @param list A list of items to be added.
   *
   * @see append(const SBase* item)
   */
  int appendFrom(const ListOf* list);
  

  /** 
   * Inserts the item at the given position of this ListOf
   * 
   * This variant of the method makes a clone of the @p item handet to it. 
   * This means that when the ListOf is destroyed, the original items will
   * not be destroyed. 
   * 
   * @param location the location where to insert the item
   * @param item the item to be inserted to the list
   * 
   * @see insertAndOwn(int location, SBase* item)
   */
  int insert(int location, const SBase* item);


  /** 
   * Inserts the item at the given position of this ListOf
   * 
   * This variant of the method makes a clone of the @p item handet to it. 
   * This means that when the ListOf is destroyed, the original items will
   * not be destroyed. 
   * 
   * @param location the location where to insert the item
   * @param item the item to be inserted to the list
   * 
   * @see insert(int location, const SBase* item)
   */
  int insertAndOwn(int location, SBase* item);


  /**
   * Get an item from the list.
   *
   * @param n the index number of the item to get.
   * 
   * @return the nth item in this ListOf items.
   *
   * @see size()
   */
  virtual const SBase* get (unsigned int n) const;


  /**
   * Get an item from the list.
   *
   * @param n the index number of the item to get.
   * 
   * @return the nth item in this ListOf items.
   *
   * @see size()
   */
  virtual SBase* get (unsigned int n);


  /**
   * Returns the first child element found that has the given id in the model-wide SId namespace, or NULL if no such object is found.
   *
   * @param id string representing the id of objects to find
   *
   * @return pointer to the first element found with the given id.
   */
  virtual SBase* getElementBySId(std::string id);
  
  
  /**
   * Returns the first child element it can find with the given metaid, or NULL if no such object is found.
   *
   * @param metaid string representing the metaid of objects to find
   *
   * @return pointer to the first element found with the given metaid.
   */
  virtual SBase* getElementByMetaId(std::string metaid);
  
  /**
   * Returns a List of all child SBase* objects, including those nested to an arbitrary depth
   *
   * @return a List* of pointers to all children objects.
   */
  virtual List* getAllElements();
  
  
#if 0
  /**
   * Get an item from the list based on its identifier.
   *
   * @param sid a string representing the the identifier of the item to get.
   * 
   * @return item in this ListOf items with the given id or @c NULL if no such
   * item exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual const SBase* get (const std::string& sid) const;
#endif


#if 0
  /**
   * Get an item from the list based on its identifier.
   *
   * @param sid a string representing the the identifier of the item to get.
   * 
   * @return item in this ListOf items with the given id or @c NULL if no such
   * item exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual SBase* get (const std::string& sid);
#endif


  /**
   * Removes all items in this ListOf object.
   *
   * If parameter @p doDelete is @c true (default), all items in this ListOf
   * object are deleted and cleared, and thus the caller doesn't have to
   * delete those items.  Otherwise, all items are just cleared from this
   * ListOf object and the caller is responsible for deleting all items.  (In
   * that case, pointers to all items should be stored elsewhere before
   * calling this function.)
   *
   * @param doDelete if @c true (default), all items are deleted and cleared.
   * Otherwise, all items are just cleared and not deleted.
   * 
   * @if notcpp @htmlinclude warn-default-args-in-docs.html @endif@~
   */ 
  void clear (bool doDelete = true);


  /**
   * Because ListOf objects typically live as object children of their parent object and not as pointer children, this function clears itself, but does not attempt to do anything else.  If a particular ListOf subclass does indeed exist as a pointer only, this function will need to be overridden.
   *
   * @return integer value indicating success/failure of the
   * function.  @if clike The value is drawn from the
   * enumeration #OperationReturnValues_t. @endif@~ The possible values
   * returned by this function are:
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_SUCCESS LIBSBML_OPERATION_SUCCESS @endlink
   * @li @link OperationReturnValues_t#LIBSBML_OPERATION_FAILED LIBSBML_OPERATION_FAILED @endlink
   */
  virtual int removeFromParentAndDelete();

  /**
   * Removes the <em>n</em>th item from this ListOf items and returns a
   * pointer to it.
   *
   * The caller owns the returned item and is responsible for deleting it.
   *
   * @param n the index of the item to remove
   *
   * @see size()
   */
  virtual SBase* remove (unsigned int n);


#if 0
  /**
   * Removes item in this ListOf items with the given identifier.
   *
   * The caller owns the returned item and is responsible for deleting it.
   * If none of the items in this list have the identifier @p sid, then @c
   * NULL is returned.
   *
   * @param sid the identifier of the item to remove
   *
   * @return the item removed.  As mentioned above, the caller owns the
   * returned item.
   */
  virtual SBase* remove (const std::string& sid);
#endif


  /**
   * Get the size of this ListOf.
   * 
   * @return the number of items in this ListOf items.
   */
  unsigned int size () const;

  /** @cond doxygen-libsbml-internal */

  /**
   * Sets the parent SBMLDocument of this SBML object.
   *
   * @param d the SBMLDocument that should become the parent of this
   * ListOf.
   */
  virtual void setSBMLDocument (SBMLDocument* d);


  /**
   * Sets this SBML object to child SBML objects (if any).
   * (Creates a child-parent relationship by the parent)
   *
   * Subclasses must override this function if they define
   * one ore more child elements.
   * Basically, this function needs to be called in
   * constructor, copy constructor and assignment operator.
   *
   * @see setSBMLDocument
   * @see enablePackageInternal
   */
  virtual void connectToChild ();


  /** @endcond */

  /**
   * Returns the libSBML type code for this object, namely, @c
   * SBML_LIST_OF.
   * 
   * @if clike LibSBML attaches an identifying code to every kind of SBML
   * object.  These are known as <em>SBML type codes</em>.  The set of
   * possible type codes is defined in the enumeration #SBMLTypeCode_t.
   * The names of the type codes all begin with the characters @c
   * SBML_. @endif@if java LibSBML attaches an identifying code to every
   * kind of SBML object.  These are known as <em>SBML type codes</em>.  In
   * other languages, the set of type codes is stored in an enumeration; in
   * the Java language interface for libSBML, the type codes are defined as
   * static integer constants in the interface class {@link
   * libsbmlConstants}.  The names of the type codes all begin with the
   * characters @c SBML_. @endif@if python LibSBML attaches an identifying
   * code to every kind of SBML object.  These are known as <em>SBML type
   * codes</em>.  In the Python language interface for libSBML, the type
   * codes are defined as static integer constants in the interface class
   * @link libsbml@endlink.  The names of the type codes all begin with the
   * characters @c SBML_. @endif@if csharp LibSBML attaches an identifying
   * code to every kind of SBML object.  These are known as <em>SBML type
   * codes</em>.  In the C# language interface for libSBML, the type codes
   * are defined as static integer constants in the interface class @link
   * libsbmlcs.libsbml@endlink.  The names of the type codes all begin with
   * the characters @c SBML_. @endif@~
   *
   * @return the SBML type code for this object, or @link SBMLTypeCode_t#SBML_UNKNOWN SBML_UNKNOWN@endlink (default).
   *
   * @see getElementName()
   */
  virtual int getTypeCode () const;


  /**
   * Get the type code of the objects contained in this ListOf.
   * 
   * @if clike LibSBML attaches an identifying code to every kind of SBML
   * object.  These are known as <em>SBML type codes</em>.  The set of
   * possible type codes is defined in the enumeration #SBMLTypeCode_t.
   * The names of the type codes all begin with the characters @c
   * SBML_. @endif@if java LibSBML attaches an identifying code to every
   * kind of SBML object.  These are known as <em>SBML type codes</em>.  In
   * other languages, the set of type codes is stored in an enumeration; in
   * the Java language interface for libSBML, the type codes are defined as
   * static integer constants in the interface class {@link
   * libsbmlConstants}.  The names of the type codes all begin with the
   * characters @c SBML_. @endif@if python LibSBML attaches an identifying
   * code to every kind of SBML object.  These are known as <em>SBML type
   * codes</em>.  In the Python language interface for libSBML, the type
   * codes are defined as static integer constants in the interface class
   * @link libsbml@endlink.  The names of the type codes all begin with the
   * characters @c SBML_. @endif@if csharp LibSBML attaches an identifying
   * code to every kind of SBML object.  These are known as <em>SBML type
   * codes</em>.  In the C# language interface for libSBML, the type codes
   * are defined as static integer constants in the interface class @link
   * libsbmlcs.libsbml@endlink.  The names of the type codes all begin with
   * the characters @c SBML_. @endif@~
   * 
   * @return the SBML type code for the objects contained in this ListOf
   * instance, or @link SBMLTypeCode_t#SBML_UNKNOWN SBML_UNKNOWN@endlink (default).
   */
  virtual int getItemTypeCode () const;


  /**
   * Returns the XML element name of this object, which for ListOf, is
   * always @c "listOf".
   * 
   * @return the XML name of this element.
   */
  virtual const std::string& getElementName () const;


  /** @cond doxygen-libsbml-internal */
  /**
   * Subclasses should override this method to write out their contained
   * SBML objects as XML elements.  Be sure to call your parents
   * implementation of this method as well.
   */
  virtual void writeElements (XMLOutputStream& stream) const;


  /**
   * Enables/Disables the given package with this element and child
   * elements (if any).
   * (This is an internal implementation for enablePackage function)
   *
   * @note Subclasses of the SBML Core package in which one or more child
   * elements are defined must override this function.
   */
  virtual void enablePackageInternal(const std::string& pkgURI, const std::string& pkgPrefix, bool flag);
  /** @endcond */

protected:
  /** @cond doxygen-libsbml-internal */

  typedef std::vector<SBase*>           ListItem;
  typedef std::vector<SBase*>::iterator ListItemIter;

  /**
   * Subclasses should override this method to get the list of
   * expected attributes.
   * This function is invoked from corresponding readAttributes()
   * function.
   */
  virtual void addExpectedAttributes(ExpectedAttributes& attributes);

  
  /**
   * Subclasses should override this method to read values from the given
   * XMLAttributes set into their specific fields.  Be sure to call your
   * parents implementation of this method as well.
   */
  virtual void readAttributes (const XMLAttributes& attributes,
                               const ExpectedAttributes& expectedAttributes);

  /**
   * Subclasses should override this method to write their XML attributes
   * to the XMLOutputStream.  Be sure to call your parents implementation
   * of this method as well.  For example:
   *
   *   SBase::writeAttributes(stream);
   *   stream.writeAttribute( "id"  , mId   );
   *   stream.writeAttribute( "name", mName );
   *   ...
   */
  virtual void writeAttributes (XMLOutputStream& stream) const;

  virtual bool isValidTypeForList(SBase * item) {return item->getTypeCode() == getItemTypeCode();}

  ListItem mItems;

  /** @endcond */
};

LIBSBML_CPP_NAMESPACE_END

#endif  /* __cplusplus */


#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS


/**
 * Creates a new ListOf.
 *
 * @return a pointer to created ListOf.
 */
LIBSBML_EXTERN
ListOf_t *
ListOf_create (unsigned int level, unsigned int version);

/**
 * Frees the given ListOf and its constituent items.
 *
 * This function assumes each item in the list is derived from SBase.
 */
LIBSBML_EXTERN
void
ListOf_free (ListOf_t *lo);

/**
 * @return a (deep) copy of this ListOf items.
 */
LIBSBML_EXTERN
ListOf_t *
ListOf_clone (const ListOf_t *lo);


/**
 * Adds a copy of item to the end of this ListOf items.
 */
LIBSBML_EXTERN
int
ListOf_append (ListOf_t *lo, const SBase_t *item);

/**
 * Adds the given item to the end of this ListOf items.
 */
LIBSBML_EXTERN
int
ListOf_appendAndOwn (ListOf_t *lo, SBase_t *item);

/**
 * Adds clones of the given items from the second list to the end of this ListOf items.
 */
LIBSBML_EXTERN
int
ListOf_appendFrom (ListOf_t *lo, ListOf_t *list);

/**
 * inserts a copy of item to this ListOf items at the given position.
 */
LIBSBML_EXTERN
int
ListOf_insert (ListOf_t *lo, int location, const SBase_t *item);

/**
 * inserts the item to this ListOf items at the given position.
 */
LIBSBML_EXTERN
int
ListOf_insertAndOwn (ListOf_t *lo, int location, SBase_t *item);


/**
 * Returns the nth item in this ListOf items.
 */
LIBSBML_EXTERN
SBase_t *
ListOf_get (ListOf_t *lo, unsigned int n);

#if (0)
/**
 * @return item in this ListOf items with the given id or @c NULL if no such
 * item exists.
 */
LIBSBML_EXTERN
SBase_t *
ListOf_getById (ListOf_t *lo, const char *sid);
#endif

/**
 * Removes all items in this ListOf object.
 */
LIBSBML_EXTERN
void
ListOf_clear (ListOf_t *lo, int doDelete);

/**
 * Removes the nth item from this ListOf items and returns a pointer to
 * it.  The caller owns the returned item and is responsible for deleting
 * it.
 */
LIBSBML_EXTERN
SBase_t *
ListOf_remove (ListOf_t *lo, unsigned int n);

#if (0)
/**
 * Removes item in this ListOf items with the given id or @c NULL if no such
 * item exists.  The caller owns the returned item and is repsonsible for
 * deleting it.
 */
LIBSBML_EXTERN
SBase_t *
ListOf_removeById (ListOf_t *lo, const char *sid);
#endif

/**
 * Returns the number of items in this ListOf items.
 */
LIBSBML_EXTERN
unsigned int
ListOf_size (const ListOf_t *lo);

/**
 * @return the int of SBML objects contained in this ListOf or
 * SBML_UNKNOWN (default).
 */
LIBSBML_EXTERN
int
ListOf_getItemTypeCode (const ListOf_t *lo);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif  /* ListOf_h */

