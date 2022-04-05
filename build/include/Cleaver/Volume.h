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

#ifndef VOLUME_H
#define VOLUME_H

#include <vector>
#include "vec3.h"
#include "ScalarField.h"


namespace Cleaver
{

class Volume
{
public:
    Volume(const std::vector<ScalarField*> &fields, int width=0, int height=0, int depth=0);
    Volume(const std::vector<ScalarField*> &fields, const vec3 &size);

    float valueAt(const vec3 &x, int material) const;
    float valueAt(float x, float y, float z, int material) const;

    void setSize(int width, int height, int depth);
    int width() const { return m_w; }
    int height() const { return m_h; }
    int depth() const { return m_d; }
    int materials() const { return m_fields.size(); }

private:
    std::vector<ScalarField*> m_fields;
    int m_w,m_h,m_d;
};

}

#endif // VOLUME_H
