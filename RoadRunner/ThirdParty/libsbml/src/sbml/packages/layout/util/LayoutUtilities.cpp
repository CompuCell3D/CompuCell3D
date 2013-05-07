/**
 * Filename    : LayoutUtilities.cpp
 * Description : Implementation of some methods used by many of the layout files.
 * Organization: European Media Laboratories Research gGmbH
 * Created     : 2007-02-14
 *
 * Copyright 2007 European Media Laboratories Research gGmbH
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
 */

#include <sbml/packages/layout/util/LayoutUtilities.h>
#include <sbml/xml/XMLAttributes.h>
#include <sbml/common/extern.h>


LIBSBML_CPP_NAMESPACE_BEGIN

LIBSBML_EXTERN
void 
addSBaseAttributes(const SBase& object,XMLAttributes& att)
{
   if(object.isSetMetaId())
   { 
     att.add("metaid",object.getMetaId());
   }
}

LIBSBML_EXTERN
void 
addGraphicalObjectAttributes(const GraphicalObject& object,XMLAttributes& att)
{
    att.add("id",object.getId());
}



// copies the attributes from source to target
// this is sued in the assignment operators and copy constructors
LIBSBML_EXTERN
void 
copySBaseAttributes(const SBase& source,SBase& target)
{
    target.setMetaId(source.getMetaId());
//    target.setId(source.getId());
//    target.setName(source.getName());
    target.setSBMLDocument(const_cast<SBMLDocument*>(source.getSBMLDocument()));
    target.setSBOTerm(source.getSBOTerm());
    if(source.isSetAnnotation())
    {
      target.setAnnotation(new XMLNode(*const_cast<SBase&>(source).getAnnotation()));
    }
    if(source.isSetNotes())
    {
      target.setNotes(new XMLNode(*const_cast<SBase&>(source).getNotes()));
    }
    if (source.getSBMLNamespaces())
    {
      target.setSBMLNamespaces(source.getSBMLNamespaces());
    }
    List* pCVTerms=target.getCVTerms();
    // first delete all the old CVTerms
    if(pCVTerms)
    {
      while(pCVTerms->getSize()>0)
      {
        CVTerm* object=static_cast<CVTerm*>(pCVTerms->remove(0));
        delete object;
      }
      // add the cloned CVTerms from source
      if(source.getCVTerms()!=NULL)
      {
          unsigned int i=0,iMax=source.getCVTerms()->getSize();
          while(i<iMax)
          {
              target.addCVTerm(static_cast<CVTerm*>(static_cast<CVTerm*>(source.getCVTerms()->get(i))->clone()));
              ++i;
          }
      }
    }
}

LIBSBML_CPP_NAMESPACE_END
