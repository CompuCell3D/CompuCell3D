/**
 * Filename    : TextGlyph.cpp
 * Description : SBML Layout TextGlyph source
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


#include <sbml/packages/layout/sbml/TextGlyph.h>
#include <sbml/packages/layout/util/LayoutUtilities.h>
#include <sbml/packages/layout/extension/LayoutExtension.h>

#include <sbml/xml/XMLNode.h>
#include <sbml/xml/XMLToken.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/xml/XMLInputStream.h>
#include <sbml/xml/XMLOutputStream.h>

LIBSBML_CPP_NAMESPACE_BEGIN

/**
 * Creates a new TextGlyph the ids of the associated GraphicalObject and
 * the originOfText are set to the empty string. The actual text is set to
 * the empty string as well.
 */  
TextGlyph::TextGlyph (unsigned int level, unsigned int version, unsigned int pkgVersion)
 : GraphicalObject(level,version,pkgVersion)
  ,mText("")
  ,mGraphicalObject("")
  ,mOriginOfText("")
{
  //
  // (NOTE) Developers don't have to invoke setSBMLNamespacesAndOwn function as follows (commentted line)
  //        in this constuctor because the function is properly invoked in the constructor of the
  //        base class (GraphicalObject).
  //

  //setSBMLNamespacesAndOwn(new LayoutPkgNamespaces(level,version,pkgVersion));  
}


/**
 * Creates a new SpeciesGlyph with the given LayoutPkgNamespaces 
 * and the id of the associated species set to the empty string.
 */        
TextGlyph::TextGlyph (LayoutPkgNamespaces* layoutns)
 : GraphicalObject(layoutns)
  ,mText("")
  ,mGraphicalObject("")
  ,mOriginOfText("")
{
  //
  // (NOTE) Developers don't have to invoke setElementNamespace function as follows (commentted line)
  //        in this constuctor because the function is properly invoked in the constructor of the
  //        base class (LineSegment).
  //

  // setElementNamespace(layoutns->getURI());

  //
  // load package extensions bound with this object (if any) 
  //
  loadPlugins(layoutns);
}


/**
 * Creates a new TextGlpyh. The id is given as the first argument.
 */ 
TextGlyph::TextGlyph (LayoutPkgNamespaces* layoutns, const std::string& id)
 : GraphicalObject(layoutns, id)
  ,mText("")
  ,mGraphicalObject("")
  ,mOriginOfText("")
{
  //
  // (NOTE) Developers don't have to invoke setElementNamespace function as follows (commentted line)
  //        in this constuctor because the function is properly invoked in the constructor of the
  //        base class (LineSegment).
  //

  // setElementNamespace(layoutns->getURI());

  //
  // load package extensions bound with this object (if any) 
  //
  loadPlugins(layoutns);
}

/**
 * Creates a new TextGlpyh. The id is given as the first argument, the text
 * to be displayed as the second.  All other attirbutes are set to the
 * empty string.
 */ 
TextGlyph::TextGlyph (LayoutPkgNamespaces* layoutns, const std::string& id, const std::string& text)
 : GraphicalObject(layoutns, id)
  ,mText(text)
  ,mGraphicalObject("")
  ,mOriginOfText("")
{
  //
  // (NOTE) Developers don't have to invoke setElementNamespace function as follows (commentted line)
  //        in this constuctor because the function is properly invoked in the constructor of the
  //        base class (LineSegment).
  //

  // setElementNamespace(layoutns->getURI());

  //
  // load package extensions bound with this object (if any) 
  //
  loadPlugins(layoutns);
}

/**
 * Creates a new TextGlyph from the given XMLNode
 */
TextGlyph::TextGlyph(const XMLNode& node, unsigned int l2version)
 : GraphicalObject(2, l2version)
  ,mText("")
  ,mGraphicalObject("")
  ,mOriginOfText("")
{
    const XMLAttributes& attributes=node.getAttributes();
    const XMLNode* child;
    //ExpectedAttributes ea(getElementName());
    ExpectedAttributes ea;
    addExpectedAttributes(ea);
    this->readAttributes(attributes,ea);
    unsigned int n=0,nMax = node.getNumChildren();
    while(n<nMax)
    {
        child=&node.getChild(n);
        const std::string& childName=child->getName();
        if(childName=="boundingBox")
        {
            this->mBoundingBox=BoundingBox(*child);
        }
        else if(childName=="annotation")
        {
            this->mAnnotation=new XMLNode(*child);
        }
        else if(childName=="notes")
        {
            this->mNotes=new XMLNode(*child);
        }
        else
        {
            //throw;
        }
        ++n;
    }    
}

/**
 * Copy constructor.
 */
TextGlyph::TextGlyph(const TextGlyph& source):GraphicalObject(source)
{
    this->mText=source.getText();
    this->mOriginOfText=source.getOriginOfTextId();
    this->mGraphicalObject=source.getGraphicalObjectId();    
}

/**
 * Assignment operator.
 */
TextGlyph& TextGlyph::operator=(const TextGlyph& source)
{
  if(&source!=this)
  {
    GraphicalObject::operator=(source);
    this->mText=source.getText();
    this->mOriginOfText=source.getOriginOfTextId();
    this->mGraphicalObject=source.getGraphicalObjectId();    
  }
  
  return *this;
}

/**
 * Destructor.
 */ 
TextGlyph::~TextGlyph()
{
} 


/**
 * Returns the text to be displayed by the text glyph.
 */ 
const std::string&
TextGlyph::getText() const
{
  return this->mText;
}


/**
 * Sets the text to be displayed by the text glyph.
 */ 
void
TextGlyph::setText (const std::string& text)
{
  this->mText = text;
} 


/**
 * Returns the id of the associated graphical object.
 */ 
const std::string&
TextGlyph::getGraphicalObjectId () const
{
  return this->mGraphicalObject;
}


/**
 * Sets the id of the associated graphical object.
 */ 
int
TextGlyph::setGraphicalObjectId (const std::string& id)
{
  return SyntaxChecker::checkAndSetSId(id,mGraphicalObject);
}


/**
 * Returns the id of the origin of text.
 */ 
const std::string&
TextGlyph::getOriginOfTextId () const
{
  return this->mOriginOfText;
}


/**
 * Sets the id of the origin of text.
 */ 
int
TextGlyph::setOriginOfTextId (const std::string& orig)
{
  return SyntaxChecker::checkAndSetSId(orig,mOriginOfText);
}


/**
 * Returns true if the text is not the empty string.
 */ 
bool
TextGlyph::isSetText () const
{
  return ! this->mText.empty();
}


/**
 * Returns true if the id of the origin of text is not the empty string.
 */ 
bool
TextGlyph::isSetOriginOfTextId () const
{
  return ! this->mOriginOfText.empty();
}


/**
 * Returns true if the id of the associated graphical object is not the
 * empty string.
 */ 
bool
TextGlyph::isSetGraphicalObjectId () const
{
  return ! this->mGraphicalObject.empty();
}


/**
 * Calls initDefaults from GraphicalObject.
 */ 
void
TextGlyph::initDefaults()
{
  GraphicalObject::initDefaults();
}

/**
 * Subclasses should override this method to return XML element name of
 * this SBML object.
 */
const std::string& TextGlyph::getElementName () const 
{
  static const std::string name = "textGlyph";
  return name;
}

/**
 * @return a (deep) copy of this Model.
 */
TextGlyph* 
TextGlyph::clone () const
{
    return new TextGlyph(*this);
}


/**
 * @return the SBML object corresponding to next XMLToken in the
 * XMLInputStream or NULL if the token was not recognized.
 */
SBase*
TextGlyph::createObject (XMLInputStream& stream)
{
  SBase*        object = 0;

  object=GraphicalObject::createObject(stream);
  
  return object;
}

/**
 * Subclasses should override this method to get the list of
 * expected attributes.
 * This function is invoked from corresponding readAttributes()
 * function.
 */
void
TextGlyph::addExpectedAttributes(ExpectedAttributes& attributes)
{
  GraphicalObject::addExpectedAttributes(attributes);

  attributes.add("text");
  attributes.add("graphicalObject");
  attributes.add("originOfText");
}


/**
 * Subclasses should override this method to read values from the given
 * XMLAttributes set into their specific fields.  Be sure to call your
 * parents implementation of this method as well.
 */

void TextGlyph::readAttributes (const XMLAttributes& attributes,
                                const ExpectedAttributes& expectedAttributes)
{
  GraphicalObject::readAttributes(attributes,expectedAttributes);

  const unsigned int sbmlLevel   = getLevel  ();
  const unsigned int sbmlVersion = getVersion();

  bool assigned = attributes.readInto("graphicalObject", mGraphicalObject, getErrorLog(), false, getLine(), getColumn());
  if (assigned && mGraphicalObject.empty())
  {
    logEmptyString(mGraphicalObject, sbmlLevel, sbmlVersion, "<" + getElementName() + ">");
  }
  if (!SyntaxChecker::isValidInternalSId(mGraphicalObject)) logError(InvalidIdSyntax);

  assigned = attributes.readInto("originOfText", mOriginOfText, getErrorLog(), false, getLine(), getColumn());
  if (assigned && mOriginOfText.empty())
  {
    logEmptyString(mOriginOfText, sbmlLevel, sbmlVersion, "<" + getElementName() + ">");
  }
  if (!SyntaxChecker::isValidInternalSId(mOriginOfText)) logError(InvalidIdSyntax);  

  attributes.readInto("text", mText, getErrorLog(), false, getLine(), getColumn());
}

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
void TextGlyph::writeElements (XMLOutputStream& stream) const
{
  GraphicalObject::writeElements(stream);

  //
  // (EXTENSION)
  //
  SBase::writeExtensionElements(stream);
}


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
void TextGlyph::writeAttributes (XMLOutputStream& stream) const
{
  GraphicalObject::writeAttributes(stream);
  if(this->isSetText())
  {
     stream.writeAttribute("text", getPrefix(), mText);
  }
  else if(this->isSetOriginOfTextId())
  {
     stream.writeAttribute("originOfText", getPrefix(), mOriginOfText);
  }
  if(this->isSetGraphicalObjectId())
  {
    stream.writeAttribute("graphicalObject",getPrefix(),  mGraphicalObject);
  }

  //
  // (EXTENSION)
  //
  SBase::writeExtensionAttributes(stream);
}


/**
 * Returns the package type code for this object.
 */
int
TextGlyph::getTypeCode () const
{
  return SBML_LAYOUT_TEXTGLYPH;
}

/**
 * Creates an XMLNode object from this.
 */
XMLNode TextGlyph::toXML() const
{
  XMLNamespaces xmlns = XMLNamespaces();
  XMLTriple triple = XMLTriple("textGlyph", "", "");
  XMLAttributes att = XMLAttributes();
  // add the SBase Ids
  addSBaseAttributes(*this,att);
  addGraphicalObjectAttributes(*this,att);
  if(this->isSetText()) att.add("text",this->mText);
  if(this->isSetGraphicalObjectId()) att.add("graphicalObject",this->mGraphicalObject);
  if(this->isSetOriginOfTextId()) att.add("originOfText",this->mOriginOfText);
  XMLToken token = XMLToken(triple, att, xmlns); 
  XMLNode node(token);
  // add the notes and annotations
  if(this->mNotes) node.addChild(*this->mNotes);
  if(this->mAnnotation) node.addChild(*this->mAnnotation);
  // write the bounding box
  node.addChild(this->mBoundingBox.toXML());
  return node;
}





/**
 * Creates a new TextGlyph and returns the pointer to it.
 */
LIBSBML_EXTERN
TextGlyph_t *
TextGlyph_create (void)
{
  return new(std::nothrow) TextGlyph;
}


/**
 * Creates a new TextGlyph from a template.
 */
LIBSBML_EXTERN
TextGlyph_t *
TextGlyph_createFrom (const TextGlyph_t *temp)
{
  return new(std::nothrow) TextGlyph(*temp);
}


/**
 * Creates a new TextGlyph with the given id
 */
LIBSBML_EXTERN
TextGlyph_t *
TextGlyph_createWith (const char *sid)
{
  LayoutPkgNamespaces layoutns;
  return new(std::nothrow) TextGlyph(&layoutns, sid ? sid : "", "");
}


/**
 * Creates a new TextGlyph referencing the give text.
 */
LIBSBML_EXTERN
TextGlyph_t *
TextGlyph_createWithText (const char *id, const char *text)
{  
  LayoutPkgNamespaces layoutns;
  return new(std::nothrow) TextGlyph(&layoutns, id ? id : "", text ? text : "");
}


/**
 * Frees the memory taken by the given text glyph.
 */
LIBSBML_EXTERN
void
TextGlyph_free (TextGlyph_t *tg)
{
  delete tg;
}


/**
 * Sets the text for the text glyph.
 */
LIBSBML_EXTERN
void
TextGlyph_setText (TextGlyph_t *tg, const char *text)
{
    tg->setText( text ? text : "" );
}


/**
 * Sets the id of the origin of the text for the text glyph.  This can be
 * the id of any valid sbml model object. The name of the object is then
 * taken as the text for the TextGlyph.
 */
LIBSBML_EXTERN
void
TextGlyph_setOriginOfTextId (TextGlyph_t *tg, const char *sid)
{
    tg->setOriginOfTextId( sid ? sid : "" );
}


/**
 * Sets the assoziated GraphicalObject id for the text glyph.  A TextGlyph
 * which is assoziated with a GraphicalObject can be considered as a label
 * to that object and they might for example be moved together in an
 * editor.
 */
LIBSBML_EXTERN
void
TextGlyph_setGraphicalObjectId (TextGlyph_t *tg, const char *sid)
{
    tg->setGraphicalObjectId( sid ? sid : "" );
}


/**
 * Returns the text associated with this text glyph.
 */
LIBSBML_EXTERN
const char *
TextGlyph_getText (const TextGlyph_t *tg)
{
    return tg->isSetText() ? tg->getText().c_str() : NULL;
}


/**
 * Returns the id of the origin of the text associated with this text
 * glyph.
 */
LIBSBML_EXTERN
const char *
TextGlyph_getGraphicalObjectId (const TextGlyph_t *tg)
{
    return tg->isSetGraphicalObjectId() ? tg->getGraphicalObjectId().c_str() : NULL;
}


/**
 * Returns the id of the graphical object associated with this text glyph.
 */
LIBSBML_EXTERN
const char *
TextGlyph_getOriginOfTextId (const TextGlyph_t *tg)
{
    return tg->isSetOriginOfTextId() ? tg->getOriginOfTextId().c_str() : NULL;
}


/**
 * Returns true is the text attribute is not the empty string.
 */
LIBSBML_EXTERN
int
TextGlyph_isSetText (const TextGlyph_t *tg)
{
  return static_cast<int>( tg->isSetText() );
}


/**
 * Returns true is the originOfText attribute is not the empty string.
 */
LIBSBML_EXTERN
int
TextGlyph_isSetOriginOfTextId (const TextGlyph_t *tg)
{
  return static_cast<int>( tg->isSetOriginOfTextId() );
}


/**
 * Returns true is the id of the associated graphical object is not the
 * empty string.
 */
LIBSBML_EXTERN
int
TextGlyph_isSetGraphicalObjectId (const TextGlyph_t *tg)
{
  return static_cast<int>( tg->isSetGraphicalObjectId() );
}


/**
 * Calls initDefaults from GraphicalObject.
 */ 
LIBSBML_EXTERN
void
TextGlyph_initDefaults (TextGlyph_t *tg)
{
  tg->initDefaults();
}

/**
 * @return a (deep) copy of this Model.
 */
LIBSBML_EXTERN
TextGlyph_t *
TextGlyph_clone (const TextGlyph_t *m)
{
  return static_cast<TextGlyph*>( m->clone() );
}

LIBSBML_CPP_NAMESPACE_END

