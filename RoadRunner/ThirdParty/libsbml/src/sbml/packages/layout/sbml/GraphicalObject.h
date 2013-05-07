/**
 * Filename    : GraphicalObject.h
 * Description : SBML Layout GraphicalObject C++ Header
 * Organization: European Media Laboratories Research gGmbH
 * Created     : 2004-07-15
 *
 * Copyright 2004 European Media Laboratories Research gGmbH
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation; either version 2.1 of the License, or
 * any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY, WITHOUT EVEN THE IMPLIED WARRANTY OF
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.  The software and
 * documentation provided hereunder is on an "as is" basis, and the
 * European Media Laboratories Research gGmbH have no obligations to
 * provide maintenance, support, updates, enhancements or modifications.
 * In no event shall the European Media Laboratories Research gGmbH be
 * liable to any party for direct, indirect, special, incidental or
 * consequential damages, including lost profits, arising out of the use of
 * this software and its documentation, even if the European Media
 * Laboratories Research gGmbH have been advised of the possibility of such
 * damage.  See the GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this library; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * The original code contained here was initially developed by:
 *
 *     Ralph Gauges
 *     Bioinformatics Group
 *     European Media Laboratories Research gGmbH
 *     Schloss-Wolfsbrunnenweg 31c
 *     69118 Heidelberg
 *     Germany
 *
 *     http://www.eml-research.de/english/Research/BCB/
 *     mailto:ralph.gauges@eml-r.villa-bosch.de
 *
 * Contributor(s):
 *
 *     Akiya Jouraku <jouraku@bio.keio.ac.jp>
 *     Modified this file for package extension in libSBML5
 *
 */


#ifndef GraphicalObject_H__
#define GraphicalObject_H__


#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>
#include <sbml/packages/layout/common/layoutfwd.h>


#ifdef __cplusplus


#include <string>

#include <sbml/SBase.h>
#include <sbml/packages/layout/sbml/BoundingBox.h>
#include <sbml/packages/layout/extension/LayoutExtension.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class LIBSBML_EXTERN GraphicalObject : public SBase
{
protected:

  std::string mId;
  std::string mMetaIdRef;

  BoundingBox mBoundingBox;
        

public:

  /**
   * Creates a new GraphicalObject.
   */
  
  GraphicalObject (unsigned int level      = LayoutExtension::getDefaultLevel(),
                   unsigned int version    = LayoutExtension::getDefaultVersion(),
                   unsigned int pkgVersion = LayoutExtension::getDefaultPackageVersion());


  /**
   * Creates a new GraphicalObject with the given LayoutPkgNamespaces
   */
  GraphicalObject (LayoutPkgNamespaces* layoutns);

  /**
   * Creates a new GraphicalObject with the given id.
   *
   * (FOR BACKWARD COMPATIBILITY)
   *
   */
  
  GraphicalObject (LayoutPkgNamespaces* layoutns, const std::string& id);

  /**
   * Creates a new GraphicalObject with the given id and 2D coordinates for
   * the bounding box.
   *
   * (FOR BACKWARD COMPATIBILITY)
   *
   */
  
  GraphicalObject (LayoutPkgNamespaces* layoutns, const std::string& id,
                   double x, double y, double w, double h);

  /**
   * Creates a new GraphicalObject with the given id and 3D coordinates for
   * the bounding box.
   *
   * (FOR BACKWARD COMPATIBILITY)
   *
   */
  
  GraphicalObject (LayoutPkgNamespaces* layoutns, const std::string& id,
                   double x, double y, double z,
                   double w, double h, double d);

  /**
   * Creates a new GraphicalObject with the given id and 3D coordinates for
   * the bounding box.
   *
   * (FOR BACKWARD COMPATIBILITY)
   *
   */
  
  GraphicalObject (LayoutPkgNamespaces* layoutns, const std::string& id, const Point* p, const Dimensions* d);

  /**
   * Creates a new GraphicalObject with the given id and 3D coordinates for
   * the bounding box.
   *
   * (FOR BACKWARD COMPATIBILITY)
   *
   */
  
  GraphicalObject (LayoutPkgNamespaces* layoutns, const std::string& id, const BoundingBox* bb);


  /**
   * Creates a new GraphicalObject from the given XMLNode
   *
   * (FOR BACKWARD COMPATIBILITY)
   *
   */
  GraphicalObject(const XMLNode& node, unsigned int l2version=4);

  /**
   * Copy constructor.
   */
   GraphicalObject(const GraphicalObject& source);

  /**
   * Assignment operator.
   */
   virtual GraphicalObject& operator=(const GraphicalObject& source);


  /**
   * Destructor.
   */ 
  
  virtual ~GraphicalObject ();

  /**
   * Does nothing. No defaults are defined for GraphicalObject.
   */ 
  
  void initDefaults ();

  /**
   * Returns the value of the "id" attribute of this GraphicalObject.
   */
  virtual const std::string& getId () const;


  /**
   * Predicate returning @c true or @c false depending on whether this
   * GraphicalObject's "id" attribute has been set.
   */
  virtual bool isSetId () const;

  
  /**
   * Sets the value of the "id" attribute of this GraphicalObject.
   */
  virtual int setId (const std::string& id);


  /**
   * Unsets the value of the "id" attribute of this GraphicalObject.
   */
  virtual int unsetId ();

  /**
   * Returns the value of the "metaidRef" attribute of this GraphicalObject.
   */
  virtual const std::string& getMetaIdRef () const;


  /**
   * Predicate returning @c true or @c false depending on whether this
   * GraphicalObject's "metaidRef" attribute has been set.
   */
  virtual bool isSetMetaIdRef () const;

  
  /**
   * Sets the value of the "metaidRef" attribute of this GraphicalObject.
   */
  virtual int setMetaIdRef (const std::string& metaid);


  /**
   * Unsets the value of the "metaidRef" attribute of this GraphicalObject.
   */
  virtual int unsetMetaIdRef ();
  
  /**
   * Sets the boundingbox for the GraphicalObject.
   */ 
  
  void setBoundingBox (const BoundingBox* bb);

  /**
   * Returns the bounding box for the GraphicalObject.
   */ 
  
  BoundingBox* getBoundingBox ();

  /**
   * Returns the bounding box for the GraphicalObject.
   */ 
  
  const BoundingBox* getBoundingBox() const;

  /**
   * Subclasses should override this method to write out their contained
   * SBML objects as XML elements.  Be sure to call your parents
   * implementation of this method as well.  For example:
   *
   *   SBase::writeElements(stream);
   *   mReactans.write(stream);
   *   mProducts.write(stream);
   *   ...
   */
  virtual void writeElements (XMLOutputStream& stream) const;

  /**
   * Subclasses should override this method to return XML element name of
   * this SBML object.
   */
  virtual const std::string& getElementName () const ;

  /**
   * @return a (deep) copy of this GraphicalObject.
   */
  virtual GraphicalObject* clone () const;


  /**
   * Returns the libSBML type code for this object.
   *
   * This method MAY return the typecode of this SBML object or it MAY
   * return SBML_UNKNOWN.  That is, subclasses of SBase are not required to
   * implement this method to return a typecode.  This method is meant
   * primarily for the LibSBML C interface where class and subclass
   * information is not readily available.
   *
   * @note In libSBML 5, the type of return value has been changed from
   *       SBMLTypeCode_t to int. The return value is one of enum values defined
   *       for each package. For example, return values will be one of
   *       SBMLTypeCode_t if this object is defined in SBML core package,
   *       return values will be one of SBMLLayoutTypeCode_t if this object is
   *       defined in Layout extension (i.e. similar enum types are defined in
   *       each pacakge extension for each SBase subclass)
   *       The value of each typecode can be duplicated between those of
   *       different packages. Thus, to distinguish the typecodes of different
   *       packages, not only the return value of getTypeCode() but also that of
   *       getPackageName() must be checked.
   *
   * @return the typecode (int value) of this SBML object or SBML_UNKNOWN
   * (default).
   *
   * @see getElementName()
   * @see getPackageName()
   */
  virtual int getTypeCode () const;


  /**
   * Accepts the given SBMLVisitor.
   *
   * @return the result of calling <code>v.visit()</code>, which indicates
   * whether or not the Visitor would like to visit the SBML object's next
   * sibling object (if available).
   */
  virtual bool accept (SBMLVisitor& v) const;
    

   /**
    * Creates an XMLNode object from this.
    */
    virtual XMLNode toXML() const;


  /** @cond doxygen-libsbml-internal */
  /**
   * Sets the parent SBMLDocument of this SBML object.
   *
   * @param d the SBMLDocument object to use
   */
  virtual void setSBMLDocument (SBMLDocument* d);


  /**
   * Sets this SBML object to child SBML objects (if any).
   * (Creates a child-parent relationship by the parent)
   *
   * Subclasses must override this function if they define
   * one ore more child elements.
   * Basically, this function needs to be called in
   * constructor, copy constructor, assignment operator.
   *
   * @see setSBMLDocument
   * @see enablePackageInternal
   */
  virtual void connectToChild ();


  /**
   * Enables/Disables the given package with this element and child
   * elements (if any).
   * (This is an internal implementation for enablePakcage function)
   *
   * @note Subclasses in which one or more child elements are defined
   * must override this function.
   */
  virtual void enablePackageInternal(const std::string& pkgURI,
                                     const std::string& pkgPrefix, bool flag);
  /** @endcond */
    
protected:
  /**
   * @return the SBML object corresponding to next XMLToken in the
   * XMLInputStream or NULL if the token was not recognized.
   */
  virtual SBase*
  createObject (XMLInputStream& stream);

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


};


class LIBSBML_EXTERN ListOfGraphicalObjects : public ListOf
{
public:

  /**
   * @return a (deep) copy of this ListOfGraphicalObjects.
   */
  virtual ListOfGraphicalObjects* clone () const;


  /**
   * Ctor.
   */
   ListOfGraphicalObjects(unsigned int level      = LayoutExtension::getDefaultLevel(),
                          unsigned int version    = LayoutExtension::getDefaultVersion(),
                          unsigned int pkgVersion = LayoutExtension::getDefaultPackageVersion());


  /**
   * Ctor.
   */
   ListOfGraphicalObjects(LayoutPkgNamespaces* layoutns);


  /**
   * @return the const char* of SBML objects contained in this ListOf or
   * SBML_UNKNOWN (default).
   */
  virtual int getItemTypeCode () const;

  /**
   * Subclasses should override this method to return XML element name of
   * this SBML object.
   */
  virtual const std::string& getElementName () const;

  /* 
   * Allow overwriting the element name (as used by the generalGlyph)
   */ 
  void setElementName(const std::string& elementName);

  /**
   * Get a GraphicalObject from the ListOfGraphicalObjects.
   *
   * @param n the index number of the GraphicalObject to get.
   * 
   * @return the nth GraphicalObject in this ListOfGraphicalObjects.
   *
   * @see size()
   */
  virtual GraphicalObject * get(unsigned int n); 


  /**
   * Get a GraphicalObject from the ListOfGraphicalObjects.
   *
   * @param n the index number of the GraphicalObject to get.
   * 
   * @return the nth GraphicalObject in this ListOfGraphicalObjects.
   *
   * @see size()
   */
  virtual const GraphicalObject * get(unsigned int n) const; 

  /**
   * Get a GraphicalObject from the ListOfGraphicalObjects
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the GraphicalObject to get.
   * 
   * @return GraphicalObject in this ListOfGraphicalObjects
   * with the given id or NULL if no such
   * GraphicalObject exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual GraphicalObject* get (const std::string& sid);


  /**
   * Get a GraphicalObject from the ListOfGraphicalObjects
   * based on its identifier.
   *
   * @param sid a string representing the identifier 
   * of the GraphicalObject to get.
   * 
   * @return GraphicalObject in this ListOfGraphicalObjects
   * with the given id or NULL if no such
   * GraphicalObject exists.
   *
   * @see get(unsigned int n)
   * @see size()
   */
  virtual const GraphicalObject* get (const std::string& sid) const;


  /**
   * Removes the nth item from this ListOfGraphicalObjects items and returns a pointer to
   * it.
   *
   * The caller owns the returned item and is responsible for deleting it.
   *
   * @param n the index of the item to remove
   *
   * @see size()
   */
  virtual GraphicalObject* remove (unsigned int n);


  /**
   * Removes item in this ListOfGraphicalObjects items with the given identifier.
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
  virtual GraphicalObject* remove (const std::string& sid);


   /**
    * Creates an XMLNode object from this.
    */
    XMLNode toXML() const;
    
protected:

  /**
   * @return the SBML object corresponding to next XMLToken in the
   * XMLInputStream or NULL if the token was not recognized.
   */
  virtual SBase* createObject (XMLInputStream& stream);
  
  virtual bool isValidTypeForList(SBase * item);

  
  std::string mElementName;

};



LIBSBML_CPP_NAMESPACE_END

#endif /* __cplusplus */


#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS



/**
 * Creates a new GraphicalObject.
 */
LIBSBML_EXTERN
GraphicalObject_t *
GraphicalObject_create (void);


/**
 * Creates a GraphicalObject from a template.
 */
LIBSBML_EXTERN
GraphicalObject_t *
GraphicalObject_createFrom (const GraphicalObject_t *temp);

/**
 * Frees all memory taken up by the GraphicalObject.
 */ 
LIBSBML_EXTERN
void
GraphicalObject_free (GraphicalObject_t *go);


/**
 * Sets the boundingbox for the GraphicalObject.
 */ 
LIBSBML_EXTERN
void
GraphicalObject_setBoundingBox (GraphicalObject_t *go, const BoundingBox_t *bb);

/**
 * Returns the bounding box for the GraphicalObject.
 */ 
LIBSBML_EXTERN
BoundingBox_t *
GraphicalObject_getBoundingBox (GraphicalObject_t *go);

/**
 * Does nothing. No defaults are defined for GraphicalObject.
 */ 
LIBSBML_EXTERN
void
GraphicalObject_initDefaults (GraphicalObject_t *go);

/**
 * @return a (deep) copy of this Model.
 */
LIBSBML_EXTERN
GraphicalObject_t *
GraphicalObject_clone (const GraphicalObject_t *m);


LIBSBML_EXTERN
int
GraphicalObject_isSetId (const GraphicalObject_t *go);

LIBSBML_EXTERN
const char *
GraphicalObject_getId (const GraphicalObject_t *go);


LIBSBML_EXTERN
int
GraphicalObject_setId (GraphicalObject_t *go, const char *sid);


LIBSBML_EXTERN
void
GraphicalObject_unsetId (GraphicalObject_t *go);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END

#endif  /* !SWIG */
#endif  /* GraphicalObject_H__ */
