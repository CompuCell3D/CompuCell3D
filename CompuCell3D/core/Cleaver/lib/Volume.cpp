//-------------------------------------------------------------------
//-------------------------------------------------------------------
//
// Cleaver - A MultiMaterial Conforming Tetrahedral Meshing Library
//
// -- Volume Class
//
// Author: Jonathan Bronson (bronson@sci.utah.ed)
//
//-------------------------------------------------------------------
//-------------------------------------------------------------------
//
//  Copyright (C) 2011, 2012, Jonathan Bronson
//  Scientific Computing & Imaging Institute
//  University of Utah
//
//  Permission is  hereby  granted, free  of charge, to any person
//  obtaining a copy of this software and associated documentation
//  files  ( the "Software" ),  to  deal in  the  Software without
//  restriction, including  without limitation the rights to  use,
//  copy, modify,  merge, publish, distribute, sublicense,  and/or
//  sell copies of the Software, and to permit persons to whom the
//  Software is  furnished  to do  so,  subject  to  the following
//  conditions:
//
//  The above  copyright notice  and  this permission notice shall
//  be included  in  all copies  or  substantial  portions  of the
//  Software.
//
//  THE SOFTWARE IS  PROVIDED  "AS IS",  WITHOUT  WARRANTY  OF ANY
//  KIND,  EXPRESS OR IMPLIED, INCLUDING  BUT NOT  LIMITED  TO THE
//  WARRANTIES   OF  MERCHANTABILITY,  FITNESS  FOR  A  PARTICULAR
//  PURPOSE AND NONINFRINGEMENT. IN NO EVENT  SHALL THE AUTHORS OR
//  COPYRIGHT HOLDERS  BE  LIABLE FOR  ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
//  ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
//  USE OR OTHER DEALINGS IN THE SOFTWARE.
//-------------------------------------------------------------------
//-------------------------------------------------------------------

#include "Volume.h"
#include "BoundingBox.h"

namespace Cleaver
{

Volume::Volume(const std::vector<ScalarField*> &fields, int width, int height, int depth) :
    m_fields(fields), m_w(width), m_h(height), m_d(depth)
{
    if(m_fields.size() > 0)
    {
        if(m_w == 0)
            m_w = m_fields[0]->bounds().size.x;
        if(m_h == 0)
            m_h = m_fields[0]->bounds().size.y;
        if(m_d == 0)
            m_d = m_fields[0]->bounds().size.z;

    }
}



Volume::Volume(const std::vector<ScalarField*> &fields, const vec3 &size) :
    m_fields(fields), m_w(size.x), m_h(size.y), m_d(size.z)
{
    if(m_fields.size() > 0)
    {
        if(m_w == 0)
            m_w = m_fields[0]->bounds().size.x;
        if(m_h == 0)
            m_h = m_fields[0]->bounds().size.y;
        if(m_d == 0)
            m_d = m_fields[0]->bounds().size.z;

    }
}

void Volume::setSize(int width, int height, int depth)
{
    m_w = width;
    m_h = height;
    m_d = depth;
}

float Volume::valueAt(const vec3 &x, int material) const
{    
    vec3 tx = vec3(m_fields[material]->bounds().size.x*(x.x / m_w),
                   m_fields[material]->bounds().size.y*(x.y / m_h),
                   m_fields[material]->bounds().size.z*(x.z / m_d));
    return m_fields[material]->valueAt(tx);
}

float Volume::valueAt(float x, float y, float z, int material) const
{
    vec3 tx = vec3(m_fields[material]->bounds().size.x*(x / m_w),
                   m_fields[material]->bounds().size.y*(y / m_h),
                   m_fields[material]->bounds().size.z*(z / m_d));
    return m_fields[material]->valueAt(tx);
}

}
