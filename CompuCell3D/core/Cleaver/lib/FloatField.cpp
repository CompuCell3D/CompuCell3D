//-------------------------------------------------------------------
//-------------------------------------------------------------------
//
// Cleaver - A MultiMaterial Tetrahedral Mesher
// -- 3D Float Point Data Field
//
//  Author: Jonathan Bronson (bronson@sci.utah.edu)
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

#include "FloatField.h"
#include <cmath>

namespace Cleaver
{

FloatField::FloatField(int width=0, int height=0, int depth = 0, float *data = 0) : m_bounds(vec3::zero, vec3(width,height,depth)), m_data(data)
{
    // no allocation
}

FloatField::~FloatField()
{
    // no memory cleanup
}

float* FloatField::data() const
{
    return m_data;
}

void FloatField::setBounds(const BoundingBox &bounds)
{
    m_bounds = bounds;
}

BoundingBox FloatField::bounds() const
{
    return m_bounds;
}

float FloatField::valueAt(float x, float y, float z) const
{
    int w = m_bounds.size.x;
    int h = m_bounds.size.y;

    // return array value if on the grid
    if(fmod(x,1.0f) == 0 && fmod(y,1.0f) == 0 && fmod(z,1.0f) == 0)
        return m_data[(int)(x + y*w + z*w*h)];

    // otherwise interpolate
    else
    {
        int i = (int)x;
        int j = (int)y;
        int k = (int)z;

        float t = fmod(x,1.0f);
        float u = fmod(y,1.0f);
        float v = fmod(z,1.0f);

        double C000 = m_data[i + j*w + k*w*h];
        double C001 = m_data[i + j*w + (k+1)*w*h];
        double C010 = m_data[i + (j+1)*w + k*w*h];
        double C011 = m_data[i + (j+1)*w + (k+1)*w*h];
        double C100 = m_data[i+1 + j*w + k*w*h];
        double C101 = m_data[i+1 + j*w + (k+1)*w*h];
        double C110 = m_data[i+1 + (j+1)*w + k*w*h];
        double C111 = m_data[i+1 + (j+1)*w + (k+1)*w*h];

        return float((1-t)*(1-u)*(1-v)*C000 + (1-t)*(1-u)*(v)*C001 +
                     (1-t)*  (u)*(1-v)*C010 + (1-t)*  (u)*(v)*C011 +
                     (t)*(1-u)*(1-v)*C100 +   (t)*(1-u)*(v)*C101 +
                     (t)*  (u)*(1-v)*C110 +   (t)*  (u)*(v)*C111);
    }
}

}
