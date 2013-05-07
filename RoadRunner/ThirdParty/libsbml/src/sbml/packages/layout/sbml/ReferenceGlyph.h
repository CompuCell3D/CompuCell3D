/**
 * Filename    : ReferenceGlyph.h
 * Description : SBML Layout ReferenceGlyph C++ Header
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


#ifndef ReferenceGlyph_H__
#define ReferenceGlyph_H__

#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>
#include <sbml/packages/layout/common/layoutfwd.h>

#ifdef __cplusplus


#include <string>

#include <sbml/packages/layout/sbml/GraphicalObject.h>
#include <sbml/packages/layout/sbml/Curve.h>
#include <sbml/packages/layout/extension/LayoutExtension.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class LIBSBML_EXTERN ReferenceGlyph : public GraphicalObject
{
protected:

  std::string mReference;
  std::string mGlyph;
  std::string mRole;
  Curve mCurve;
  
public:

  /**
   * Creates a new ReferenceGlyph with the given SBML level, version and
   * package version.  The id if the associated 
   * reference and the id of the associated  glyph are set to the
   * empty string.  The role is set to empty.
   */
  
  ReferenceGlyph (unsigned int level      = LayoutExtension::getDefaultLevel(),
                         unsigned int version    = LayoutExtension::getDefaultVersion(),
                         unsigned int pkgVersion = LayoutExtension::getDefaultPackageVersion());

  
  /**
   * Ctor.
   */
  ReferenceGlyph(LayoutPkgNamespaces* layoutns);
        

  /**
   * Creates a new ReferenceGlyph.  The id is given as the first
   * argument, the id of the associated reference is given as the
   * second argument.  The third argument is the id of the associated
   * glpyh and the fourth argument is the role.
   */ 
  
  ReferenceGlyph (LayoutPkgNamespaces* layoutns, const std::string& sid,
                          const std::string& referenceId,
                          const std::string& glyphId,
                          const std::string& role );
        

  /**
   * Creates a new ReferenceGlyph from the given XMLNode
   */
  ReferenceGlyph(const XMLNode& node, unsigned int l2version=4);

  /**
   * Copy constructor.
   */
   ReferenceGlyph(const ReferenceGlyph& source);

  /**
   * Assignment operator.
   */
   virtual ReferenceGlyph& operator=(const ReferenceGlyph& source);

  /**
   * Destructor.
   */ 
  
  virtual ~ReferenceGlyph (); 

        
  /**
   * Returns the id of the associated glyph.
   */ 
  
  const std::string& getGlyphId () const;
        
  /**
   * Sets the id of the associated glyph.
   */ 
  
  void setGlyphId (const std::string& glyphId);
        
  /**
   * Returns the id of the associated sbml reference.
   */ 
  
  const std::string& getReferenceId() const;
        
  /**
   * Sets the id of the associated sbml reference.
   */ 
  
  void setReferenceId (const std::string& id);

  /**
   * Returns a string representation of the role.
   */ 
  
  const std::string& getRole() const;
  
  /**
   * Sets the role.
   */ 
  
  void setRole (const std::string& role);
        
  /**
   * Returns the curve object for the reference glyph
   */ 
  Curve* getCurve () ;

  /**
   * Returns the curve object for the reference glyph
   */ 
  const Curve* getCurve () const;

  /**
   * Sets the curve object for the reference glyph.
   */ 
  
  void setCurve (const Curve* curve);
       
  /**
   * Returns true if the curve consists of one or more segments.
   */ 
  
    bool isSetCurve () const;

  /**
   * Returns true if the id of the associated glpyh is not the
   * empty string.
   */ 
  
  bool isSetGlyphId () const;
        
  /**
   * Returns true if the id of the associated reference is not the
   * empty string.
   */ 
  
  bool isSetReferenceId() const;
        
  /**
   * Returns true of role is different from the empty string.
   */ 
  
  bool isSetRole () const;
        
  /**
   * Calls initDefaults on GraphicalObject 
   */ 
  
  void initDefaults ();

  /**
   * Creates a new LineSegment object, adds it to the end of the list of
   * curve segment objects of the curve and returns a reference to the
   * newly created object.
   */
  
  LineSegment* createLineSegment ();

  /**
   * Creates a new CubicBezier object, adds it to the end of the list of
   * curve segment objects of the curve and returns a reference to the
   * newly created object.
   */
  
  CubicBezier* createCubicBezier ();

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
   * @return a (deep) copy of this ReferenceGlyph.
   */
  virtual ReferenceGlyph* clone () const;


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
   
  virtual bool accept (SBMLVisitor& v) const;
   */

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

LIBSBML_CPP_NAMESPACE_END

#endif /* __cplusplus */


#ifndef SWIG

LIBSBML_CPP_NAMESPACE_BEGIN
BEGIN_C_DECLS


/**
 * Creates a new ReferenceGlyph object and returns a pointer to it.
 */
LIBSBML_EXTERN
ReferenceGlyph_t *
ReferenceGlyph_create (void);

/**
 * Creates a new ReferenceGlyph from a template.
 */
LIBSBML_EXTERN
ReferenceGlyph_t *
ReferenceGlyph_createFrom (const ReferenceGlyph_t *temp);

/**
 * Creates a new ReferenceGlyph object with the given id and returns
 * a pointer to it.
 */
LIBSBML_EXTERN
ReferenceGlyph_t *
ReferenceGlyph_createWith ( const char *sid,
                                   const char *glyphId,
                                   const char *referenceId,
                                   const char* role );


/**
 * Frees the memory for the ReferenceGlyph
 */
LIBSBML_EXTERN
void
ReferenceGlyph_free (ReferenceGlyph_t *srg);


/**
 * Sets the reference for the glyph.
 */
LIBSBML_EXTERN
void
ReferenceGlyph_setReferenceId (ReferenceGlyph_t *srg,
                                             const char *id);

/**
 * Gets the reference id for the given  glyph.
 */
LIBSBML_EXTERN
const char *
ReferenceGlyph_getReferenceId(const ReferenceGlyph_t *);

/**
 * Returns 0 if the reference reference has not been set for this
 * glyph and 1 otherwise.
 */
LIBSBML_EXTERN
int
ReferenceGlyph_isSetReferenceId(const ReferenceGlyph_t *);

/**
 * Sets the glyph reference for the glyph.
 */
LIBSBML_EXTERN
void
ReferenceGlyph_setGlyphId (ReferenceGlyph_t *srg,
                                         const char *id);

/**
 * Gets the reference id for the given glyph.
 */
LIBSBML_EXTERN
const char *
ReferenceGlyph_getGlyphId (const ReferenceGlyph_t *srg);

/**
 * Returns 0 if the reference has not been set for this
 * glyph and 1 otherwise.
 */
LIBSBML_EXTERN
int
ReferenceGlyph_isSetGlyphId (const ReferenceGlyph_t *srg);


/**
 * Sets the curve for the reference glyph.
 */
LIBSBML_EXTERN
void
ReferenceGlyph_setCurve (ReferenceGlyph_t *srg, Curve_t *c);

/**
 * Gets the Curve for the given reference glyph.
 */
LIBSBML_EXTERN
Curve_t *
ReferenceGlyph_getCurve (ReferenceGlyph_t *srg);

/**
 * Returns true if the Curve has one or more LineSegment.
 */
LIBSBML_EXTERN
int
ReferenceGlyph_isSetCurve(ReferenceGlyph_t* srg);

/**
 * Sets the role of the reference glyph based on the string. 
 */
LIBSBML_EXTERN
void
ReferenceGlyph_setRole (ReferenceGlyph_t *srg, const char *r);

/**
 * Returns the role of the reference.
 */ 

LIBSBML_EXTERN
const char*
ReferenceGlyph_getRole(const ReferenceGlyph_t* srg);


/**
 * Returns true if the role is not empty.
 */ 
LIBSBML_EXTERN
int
ReferenceGlyph_isSetRole(const ReferenceGlyph_t *srg);

/**
 * Calls initDefaults on GraphicalObject 
 */ 
LIBSBML_EXTERN
void
ReferenceGlyph_initDefaults (ReferenceGlyph_t *srg);

/**
 * Creates a new LineSegment object, adds it to the end of the list of
 * curve segment objects of the curve and returns a reference to the newly
 * created object.
 */
LIBSBML_EXTERN
LineSegment_t *
ReferenceGlyph_createLineSegment (ReferenceGlyph_t *srg);

/**
 * Creates a new CubicBezier object, adds it to the end of the list of
 * curve segment objects of the curve and returns a reference to the newly
 * created object.
 */
LIBSBML_EXTERN
CubicBezier_t *
ReferenceGlyph_createCubicBezier (ReferenceGlyph_t *srg);

/**
 * @return a (deep) copy of this Model.
 */
LIBSBML_EXTERN
ReferenceGlyph_t *
ReferenceGlyph_clone (const ReferenceGlyph_t *m);

LIBSBML_EXTERN
int
ReferenceGlyph_isSetId (const ReferenceGlyph_t *srg);

LIBSBML_EXTERN
const char *
ReferenceGlyph_getId (const ReferenceGlyph_t *srg);


LIBSBML_EXTERN
int
ReferenceGlyph_setId (ReferenceGlyph_t *srg, const char *sid);


LIBSBML_EXTERN
void
ReferenceGlyph_unsetId (ReferenceGlyph_t *srg);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END


#endif /* !SWIG */
#endif /* ReferenceGlyph_H__ */
