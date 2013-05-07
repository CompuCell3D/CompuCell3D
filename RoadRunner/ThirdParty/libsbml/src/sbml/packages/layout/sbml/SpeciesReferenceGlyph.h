/**
 * Filename    : SpeciesReferenceGlyph.h
 * Description : SBML Layout SpeciesReferenceGlyph C++ Header
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


#ifndef SpeciesReferenceGlyph_H__
#define SpeciesReferenceGlyph_H__

#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>
#include <sbml/packages/layout/common/layoutfwd.h>

#ifdef __cplusplus


#include <string>

#include <sbml/packages/layout/sbml/GraphicalObject.h>
#include <sbml/packages/layout/sbml/SpeciesReferenceRole.h>
#include <sbml/packages/layout/sbml/Curve.h>
#include <sbml/packages/layout/extension/LayoutExtension.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class LIBSBML_EXTERN SpeciesReferenceGlyph : public GraphicalObject
{
protected:

  std::string mSpeciesReference;
  std::string mSpeciesGlyph;
  SpeciesReferenceRole_t mRole;
  Curve mCurve;

private:
  
#ifndef SWIG
  static const std::string SPECIES_REFERENCE_ROLE_STRING[];
#endif //SWIG

public:

  /**
   * Creates a new SpeciesReferenceGlyph with the given SBML level, version and
   * package version.  The id if the associated species
   * reference and the id of the associated species glyph are set to the
   * empty string.  The role is set to SPECIES_ROLE_UNDEFINED.
   */
  
  SpeciesReferenceGlyph (unsigned int level      = LayoutExtension::getDefaultLevel(),
                         unsigned int version    = LayoutExtension::getDefaultVersion(),
                         unsigned int pkgVersion = LayoutExtension::getDefaultPackageVersion());

  
  /**
   * Ctor.
   */
  SpeciesReferenceGlyph(LayoutPkgNamespaces* layoutns);
        

  /**
   * Creates a new SpeciesReferenceGlyph.  The id is given as the first
   * argument, the id of the associated species reference is given as the
   * second argument.  The third argument is the id of the associated
   * species glpyh and the fourth argument is the role.
   */ 
  
  SpeciesReferenceGlyph (LayoutPkgNamespaces* layoutns, const std::string& sid,
                          const std::string& speciesReferenceId,
                          const std::string& speciesGlyphId,
                          SpeciesReferenceRole_t role );
        

  /**
   * Creates a new SpeciesReferenceGlyph from the given XMLNode
   */
  SpeciesReferenceGlyph(const XMLNode& node, unsigned int l2version=4);

  /**
   * Copy constructor.
   */
   SpeciesReferenceGlyph(const SpeciesReferenceGlyph& source);

  /**
   * Assignment operator.
   */
   virtual SpeciesReferenceGlyph& operator=(const SpeciesReferenceGlyph& source);

  /**
   * Destructor.
   */ 
  
  virtual ~SpeciesReferenceGlyph (); 

        
  /**
   * Returns the id of the associated SpeciesGlyph.
   */ 
  
  const std::string& getSpeciesGlyphId () const;
        
  /**
   * Sets the id of the associated species glyph.
   */ 
  
  void setSpeciesGlyphId (const std::string& speciesGlyphId);
        
  /**
   * Returns the id of the associated species reference.
   */ 
  
  const std::string& getSpeciesReferenceId() const;
        
  /**
   * Sets the id of the associated species reference.
   */ 
  
  void setSpeciesReferenceId (const std::string& id);

  /**
   * Returns a string representation of the role.
   */ 
  
  const std::string& getRoleString() const;

        
  /**
   * Returns the role.
   */ 
  
  SpeciesReferenceRole_t getRole() const;
        
  /**
   * Sets the role based on a string.
   * The String can be one of:
   * SUBSTRATE
   * PRODUCT
   * SIDESUBSTRATE
   * SIDEPRODUCT
   * MODIFIER
   * ACTIVATOR
   * INHIBITOR    
   */ 
  
  void setRole (const std::string& role);

  /**
   * Sets the role.
   */ 
  
  void setRole (SpeciesReferenceRole_t role);
        
  /**
   * Returns the curve object for the species reference glyph
   */ 
  Curve* getCurve () ;

  /**
   * Returns the curve object for the species reference glyph
   */ 
  const Curve* getCurve () const;

  /**
   * Sets the curve object for the species reference glyph.
   */ 
  
  void setCurve (const Curve* curve);
       
  /**
   * Returns true if the curve consists of one or more segments.
   */ 
  
    bool isSetCurve () const;

  /**
   * Returns true if the id of the associated species glpyh is not the
   * empty string.
   */ 
  
  bool isSetSpeciesGlyphId () const;
        
  /**
   * Returns true if the id of the associated species reference is not the
   * empty string.
   */ 
  
  bool isSetSpeciesReferenceId() const;
        
  /**
   * Returns true of role is different from SPECIES_ROLE_UNDEFINED.
   */ 
  
  bool isSetRole () const;
        
  /**
   * Calls initDefaults on GraphicalObject and sets role to
   * SPECIES_ROLE_UNDEFINED.
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
   * @return a (deep) copy of this SpeciesReferenceGlyph.
   */
  virtual SpeciesReferenceGlyph* clone () const;


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
 * Creates a new SpeciesReferenceGlyph object and returns a pointer to it.
 */
LIBSBML_EXTERN
SpeciesReferenceGlyph_t *
SpeciesReferenceGlyph_create (void);

/**
 * Creates a new SpeciesReferenceGlyph from a template.
 */
LIBSBML_EXTERN
SpeciesReferenceGlyph_t *
SpeciesReferenceGlyph_createFrom (const SpeciesReferenceGlyph_t *temp);

/**
 * Creates a new SpeciesReferenceGlyph object with the given id and returns
 * a pointer to it.
 */
LIBSBML_EXTERN
SpeciesReferenceGlyph_t *
SpeciesReferenceGlyph_createWith ( const char *sid,
                                   const char *speciesGlyphId,
                                   const char *speciesReferenceId,
                                   SpeciesReferenceRole_t role );


/**
 * Frees the memory for the SpeciesReferenceGlyph
 */
LIBSBML_EXTERN
void
SpeciesReferenceGlyph_free (SpeciesReferenceGlyph_t *srg);


/**
 * Sets the reference species for the species glyph.
 */
LIBSBML_EXTERN
void
SpeciesReferenceGlyph_setSpeciesReferenceId (SpeciesReferenceGlyph_t *srg,
                                             const char *id);

/**
 * Gets the reference species id for the given species glyph.
 */
LIBSBML_EXTERN
const char *
SpeciesReferenceGlyph_getSpeciesReferenceId(const SpeciesReferenceGlyph_t *);

/**
 * Returns 0 if the reference species reference has not been set for this
 * glyph and 1 otherwise.
 */
LIBSBML_EXTERN
int
SpeciesReferenceGlyph_isSetSpeciesReferenceId(const SpeciesReferenceGlyph_t *);

/**
 * Sets the species glyph reference for the species glyph.
 */
LIBSBML_EXTERN
void
SpeciesReferenceGlyph_setSpeciesGlyphId (SpeciesReferenceGlyph_t *srg,
                                         const char *id);

/**
 * Gets the reference speciess id for the given species glyph.
 */
LIBSBML_EXTERN
const char *
SpeciesReferenceGlyph_getSpeciesGlyphId (const SpeciesReferenceGlyph_t *srg);

/**
 * Returns 0 if the reference species reference has not been set for this
 * glyph and 1 otherwise.
 */
LIBSBML_EXTERN
int
SpeciesReferenceGlyph_isSetSpeciesGlyphId (const SpeciesReferenceGlyph_t *srg);


/**
 * Sets the curve for the species reference glyph.
 */
LIBSBML_EXTERN
void
SpeciesReferenceGlyph_setCurve (SpeciesReferenceGlyph_t *srg, Curve_t *c);

/**
 * Gets the Curve for the given species reference glyph.
 */
LIBSBML_EXTERN
Curve_t *
SpeciesReferenceGlyph_getCurve (SpeciesReferenceGlyph_t *srg);

/**
 * Returns true if the Curve has one or more LineSegment.
 */
LIBSBML_EXTERN
int
SpeciesReferenceGlyph_isSetCurve(SpeciesReferenceGlyph_t* srg);

/**
 * Sets the role of the species reference glyph based on the string.  The
 * string can be one of UNDEFINED, SUBSTRATE, PRODUCT, SIDESUBSTRATE,
 * SIDEPRODUCT, MODIFIER, INHIBITOR or ACTIVATOR.  If it is none of those,
 * the role is set to SPECIES_ROLE_UNDEFINED.
 */
LIBSBML_EXTERN
void
SpeciesReferenceGlyph_setRole (SpeciesReferenceGlyph_t *srg, const char *r);

/**
 * Returns the role of the species reference.
 */ 
LIBSBML_EXTERN
SpeciesReferenceRole_t
SpeciesReferenceGlyph_getRole (const SpeciesReferenceGlyph_t *srg);
/**
 * Returns a string representation of the role of the species reference.
 */ 
LIBSBML_EXTERN
const char*
SpeciesReferenceGlyph_getRoleString(const SpeciesReferenceGlyph_t* srg);


/**
 * Returns true if the role is not SPECIES_ROLE_UNDEFINED.
 */ 
LIBSBML_EXTERN
int
SpeciesReferenceGlyph_isSetRole(const SpeciesReferenceGlyph_t *srg);

/**
 * Calls initDefaults on GraphicalObject and sets role to
 * SPECIES_ROLE_UNDEFINED.
 */ 
LIBSBML_EXTERN
void
SpeciesReferenceGlyph_initDefaults (SpeciesReferenceGlyph_t *srg);

/**
 * Creates a new LineSegment object, adds it to the end of the list of
 * curve segment objects of the curve and returns a reference to the newly
 * created object.
 */
LIBSBML_EXTERN
LineSegment_t *
SpeciesReferenceGlyph_createLineSegment (SpeciesReferenceGlyph_t *srg);

/**
 * Creates a new CubicBezier object, adds it to the end of the list of
 * curve segment objects of the curve and returns a reference to the newly
 * created object.
 */
LIBSBML_EXTERN
CubicBezier_t *
SpeciesReferenceGlyph_createCubicBezier (SpeciesReferenceGlyph_t *srg);

/**
 * @return a (deep) copy of this Model.
 */
LIBSBML_EXTERN
SpeciesReferenceGlyph_t *
SpeciesReferenceGlyph_clone (const SpeciesReferenceGlyph_t *m);

LIBSBML_EXTERN
int
SpeciesReferenceGlyph_isSetId (const SpeciesReferenceGlyph_t *srg);

LIBSBML_EXTERN
const char *
SpeciesReferenceGlyph_getId (const SpeciesReferenceGlyph_t *srg);


LIBSBML_EXTERN
int
SpeciesReferenceGlyph_setId (SpeciesReferenceGlyph_t *srg, const char *sid);


LIBSBML_EXTERN
void
SpeciesReferenceGlyph_unsetId (SpeciesReferenceGlyph_t *srg);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END


#endif /* !SWIG */
#endif /* SpeciesReferenceGlyph_H__ */
