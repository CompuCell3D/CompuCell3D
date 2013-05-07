/**
 * Filename    : CompartmentGlyph.h
 * Description : SBML Layout CompartmentGlyph C++ Header
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


#ifndef CompartmentGlyph_H__
#define CompartmentGlyph_H__

#include <sbml/common/extern.h>
#include <sbml/common/sbmlfwd.h>
#include <sbml/packages/layout/common/layoutfwd.h>


#ifdef __cplusplus 


#include <string>
#include <sbml/packages/layout/sbml/GraphicalObject.h>

LIBSBML_CPP_NAMESPACE_BEGIN

class LIBSBML_EXTERN CompartmentGlyph : public GraphicalObject
{
protected:

  std::string mCompartment;
        
  friend class LayoutHandler;


public:
        
  /**
   * Default Constructor which creates a new CompartmentGlyph.  Id and
   * associated compartment id are unset.
   */
  
  CompartmentGlyph (unsigned int level      = LayoutExtension::getDefaultLevel(),
                    unsigned int version    = LayoutExtension::getDefaultVersion(),
                    unsigned int pkgVersion = LayoutExtension::getDefaultPackageVersion());

  /**
   * Ctor.
   */
   CompartmentGlyph(LayoutPkgNamespaces* layoutns);

        
  /**
   * Constructor which creates a new CompartmentGlyph with the given id.
   *
   * (FOR BACKWARD COMPATIBILITY)
   *
   */
  
  CompartmentGlyph (LayoutPkgNamespaces* layoutns, const std::string& id);

  /**
   * Constructor which creates a new CompartmentGlyph.  Id and associated
   * compartment id are set to copies of the values given as arguments.
   *
   * (FOR BACKWARD COMPATIBILITY)
   */
  
  CompartmentGlyph (LayoutPkgNamespaces* layoutns, 
                    const std::string& id, const std::string& compartmentId);


  /**
   * Creates a new CompartmentGlyph from the given XMLNode
   *
   * (FOR BACKWARD COMPATIBILITY)
   */
   CompartmentGlyph(const XMLNode& node, unsigned int l2version=4);

  /**
   * Copy constructor.
   */
   CompartmentGlyph(const CompartmentGlyph& source);

  /**
   * Assignment operator.
   */
  virtual  CompartmentGlyph& operator=(const CompartmentGlyph& source);

  /**
   * Destructor.
   */          
  virtual ~CompartmentGlyph ();

  /**
   * Returns the id of the associated compartment.
   */        
  
  const std::string& getCompartmentId () const;
        
  /**
   * Sets the id of the associated compartment.
   */ 
  
  int setCompartmentId (const std::string& id);

  /**
   * Returns true if the id of the associated compartment is not the empty
   * string.
   */  
  
  bool isSetCompartmentId () const;
        
  /**
   * Calls initDefaults from GraphicalObject.
   */ 
  
  void initDefaults ();

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
   * @return a (deep) copy of this Model.
   */
  virtual CompartmentGlyph* clone () const;


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
    * Creates an XMLNode object from this.
    */
    virtual XMLNode toXML() const;
    
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


#endif /* __cplusplus */


#ifndef SWIG


BEGIN_C_DECLS


/**
 * Creates a new CompartmentGlyph and returns the pointer to it.
 */
LIBSBML_EXTERN
CompartmentGlyph_t *
CompartmentGlyph_create ();


/**
 * Creates a new CompartmentGlyph from a template.
 */
LIBSBML_EXTERN
CompartmentGlyph_t *
CompartmentGlyph_createFrom (const CompartmentGlyph_t *cg);

/**
 * Creates a new CompartmentGlyph with the given id
 */
LIBSBML_EXTERN
CompartmentGlyph_t *
CompartmentGlyph_createWith (const char *sid);

/**
 * Creates a new CompartmentGlyph with the given id
 */
LIBSBML_EXTERN
CompartmentGlyph_t *
CompartmentGlyph_createWithCompartmentId (const char *sid, const char *compId);

/**
 * Frees the memory taken by the given compartment glyph.
 */
LIBSBML_EXTERN
void
CompartmentGlyph_free (CompartmentGlyph_t *cg);


/**
 * Sets the reference compartment for the compartment glyph.
 */
LIBSBML_EXTERN
void
CompartmentGlyph_setCompartmentId (CompartmentGlyph_t *cg, const char *id);

/**
 * Gets the reference compartments id for the given compartment glyph.
 */
LIBSBML_EXTERN
const char *
CompartmentGlyph_getCompartmentId (const CompartmentGlyph_t *cg);

/**
 * Returns 0 if the reference compartment has not been set for this glyph
 * and 1 otherwise.
 */
LIBSBML_EXTERN
int
CompartmentGlyph_isSetCompartmentId (const CompartmentGlyph_t *cg);


/**
 * Calls initDefaults from GraphicalObject.
 */ 
LIBSBML_EXTERN
void
CompartmentGlyph_initDefaults (CompartmentGlyph_t *cg);

/**
 * @return a (deep) copy of this Model.
 */
LIBSBML_EXTERN
CompartmentGlyph_t *
CompartmentGlyph_clone (const CompartmentGlyph_t *m);


END_C_DECLS
LIBSBML_CPP_NAMESPACE_END


#endif  /* !SWIG */
#endif  /* CompartmentGlyph_H__ */
