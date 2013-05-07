/*
 * BSD Licence:
 * Copyright (c) 2002 Herbert Sauro [ hsauro@cds.caltech.edu ]
 * Systems Biology Group [ www.sys-bio.org ]
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, 
 * this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright 
 * notice, this list of conditions and the following disclaimer in the 
 * documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the <ORGANIZATION> nor the names of its contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 */


using System;
using System.Collections;

// This unit defines a Matrix class for use in the .NET framework.


// Matrix class of type double. The matrix class represents the mathematical


// object 'matrix' and therefore follows the maths convention that indexing


// starts at ONE.

namespace LibRoadRunner.Util.Unused
{
    // Descend Matrix class from BaseMatrix. BaseMatrix has column and row naming
    // logic and the row, column dimensions, and various test methods

    public class Matrix : BaseMatrix
    {
        // Use jagged array because we can optimize certain 
        // operations such as row exchange
        private SimpleComplex[][] mxx;
        public string name;

        public Matrix() : base()
        {
            r = 0;
            c = 0;
        }

        // ---------------------------------------------------------------------
        // One of the main constructors. Creates an empty matrix of size r by c
        // Usage:
        //   Matrix m = new Matrix(4,4);
        // ---------------------------------------------------------------------

        // Worth checking for negative values?
        public Matrix(int r, int c) : base()
        {
            this.r = r;
            this.c = c;

            mxx = new SimpleComplex[r][];
            for (int i = 0; i < r; i++)
            {
                mxx[i] = new SimpleComplex[c];
                for (int j = 0; j < c; j++)
                    mxx[i][j] = new SimpleComplex();
            }
            setDefaultLabels();
        }


        public Matrix(Matrix cpy) : base()
        {
            r = cpy.r;
            c = cpy.c;

            mxx = new SimpleComplex[cpy.r][];
            for (int i = 0; i < cpy.r; i++)
                mxx[i] = new SimpleComplex[cpy.c];

            for (int i = 0; i < cpy.r; i++)
                for (int j = 0; j < cpy.c; j++)
                    mxx[i][j] = cpy.mxx[i][j];
            for (int i = 0; i < cpy.r; i++)
                rowNames.Add(cpy.rowNames[i]);
            for (int i = 0; i < cpy.c; i++)
                columnNames.Add(cpy.columnNames[i]);
        }

        // ---------------------------------------------------------------------
        // Create a matrix from an array argument
        // ---------------------------------------------------------------------
        public Matrix(double[,] M)
        {
            r = M.GetLength(0);
            c = M.GetLength(1);

            mxx = new SimpleComplex[r][];
            for (int i = 0; i < r; i++)
                mxx[i] = new SimpleComplex[c];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    mxx[i][j] = new SimpleComplex(M[i, j], 0.0);
            setDefaultLabels();
        }

        // ---------------------------------------------------------------------
        // Create a matrix from a jagged array argument
        // ---------------------------------------------------------------------
        public Matrix(double[][] M)
        {
            r = M.GetLength(0);
            if (r > 0)
                c = M[0].GetLength(0);
            else c = 0;

            mxx = new SimpleComplex[r][];
            for (int i = 0; i < r; i++)
                mxx[i] = new SimpleComplex[c];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    mxx[i][j] = new SimpleComplex(M[i][j], 0.0);
            setDefaultLabels();
        }


        // ---------------------------------------------------------------------
        // Create an array from an array argument using the labels to label the matrix
        // ---------------------------------------------------------------------
        public Matrix(double[,] M, string[] rowNames, string[] columnNames)
        {
            r = M.GetLength(0);
            c = M.GetLength(1);

            mxx = new SimpleComplex[r][];
            for (int i = 0; i < r; i++)
                mxx[i] = new SimpleComplex[c];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    mxx[i][j] = new SimpleComplex(M[i, j], 0.0);
            for (int i = 0; i < r; i++)
                this.rowNames.Add(rowNames[i]);
            for (int i = 0; i < c; i++)
                this.columnNames.Add(columnNames[i]);
        }

        // ---------------------------------------------------------------------
        // Creates a matrix of size r by c filled with scalar value d
        // Usage:
        //   Matrix m = new Matrix (2,2,Math.Pi);
        // ---------------------------------------------------------------------

        public Matrix(int r, int c, double d) : base()
        {
            this.r = r;
            this.c = c;


            mxx = new SimpleComplex[r][];
            for (int i = 0; i < r; i++)
                mxx[i] = new SimpleComplex[c];
            setDefaultLabels();

            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    this[i, j] = new SimpleComplex(d, 0.0);
        }

        public SimpleComplex[][] data
        {
            get { return mxx; }
            set
            {
                mxx = value;
                r = value.GetLength(0);
                if (r > 0)
                    c = value[0].GetLength(0);
                else c = 0;
                setDefaultLabels();
            }
        }

        public SimpleComplex this[int r, int c]
        {
            get { return mxx[r][c]; }
            set { mxx[r][c] = value; }
            //return m_pData[nCol + nRow * m_NumColumns] ;
        }

        private void setDefaultLabels()
        {
            rowNames.Clear();
            columnNames.Clear();

            for (int i = 0; i < r; i++)
                rowNames.Add("R" + i.ToString());

            for (int i = 0; i < c; i++)
                columnNames.Add("C" + i.ToString());
        }


        // ---------------------------------------------------------------------
        // Resizes a matrix x to new dimension r,c. Resized matrix is returned,
        // any data in x is preserved in the resized matrix
        // Usage:
        //   Matrix m1 = Matrix.resize (m2, 6,5);
        // ---------------------------------------------------------------------

        public static Matrix resize(Matrix x, int r, int c)
        {
            var result = new Matrix(r, c);

            for (int i = 0; i < x.r; i++)
                for (int j = 0; j < x.c; j++)
                    if ((i < r) && (j < c))
                        result[i, j] = x[i, j];
            result.setDefaultLabels();
            for (int i = 1; i < x.r; i++)
                result.rowNames[i] = x.rowNames[i];
            for (int i = 1; i < x.c; i++)
                result.columnNames[i] = x.rowNames[i];
            return result;
        }


        // ---------------------------------------------------------------------
        // Nonstatic resize method
        // Usage: 
        //   m.resize (5,8);
        // ---------------------------------------------------------------------

        public void resize(int r, int c)
        {
            // if we have the desired size don't do anything
            if (this.c == c && this.r == r)
                return;

            // Create space for the new resized matrix
            var da = new Matrix(r, c);

            for (int i = 0; i < Math.Min(this.r, r); i++)
                for (int j = 0; j < Math.Min(this.c, c); j++)
                    da[i, j] = this[i, j];

            for (int i = 0; i < Math.Min(this.r, r); i++)
                da.rowNames[i] = rowNames[i];
            for (int i = 0; i < Math.Min(this.c, c); i++)
                da.columnNames[i] = columnNames[i];
            rowNames = (ArrayList) da.rowNames.Clone();
            columnNames = (ArrayList) da.columnNames.Clone();

            this.r = r;
            this.c = c;
            mxx = da.mxx;
        }


        // ---------------------------------------------------------------------
        // Make a copy of the matrix src, this matrix must exist and have the appropraite dimensions
        // Usage m1.Copy (m2);   m1 = m2;
        // ---------------------------------------------------------------------
        public void copy(Matrix src)
        {
            if ((r != src.r) || (c != src.c))
                throw (new EMatrixException("source and destination matrices must be the same size"));

            mxx = (SimpleComplex[][]) src.mxx.Clone();
            // Copy over column and row names, clear destination first then copy
            rowNames = (ArrayList) src.rowNames.Clone();
            columnNames = (ArrayList) src.columnNames.Clone();
        }

        // ---------------------------------------------------------------------
        // Agument this matrix with the matrix in the argument
        // Usage:
        //   m1.Augment (m2)
        //   m2 must be square and must have the same number of rows as m1
        // ---------------------------------------------------------------------
        public void Augment(Matrix a)
        {
            if ((r != a.r) && (! a.isSquare()))
                throw (new EMatrixException("source and destination matrices must be compatible to augment"));

            int originalSize_c = c;
            resize(r, c + a.c);
            for (int i = 0; i < r; i++)
                for (int j = 0; j < a.c; j++)
                    this[i, originalSize_c + j] = a[i, j];
            for (int i = 0; i < a.c; i++)
                columnNames[originalSize_c + i] = a.columnNames[i];
        }


        // ---------------------------------------------------------------------
        // Statics for generating predefined matrix types
        // Usage:
        //   m = Matrix.Identity();
        // ---------------------------------------------------------------------

        public static Matrix Identity(int n)
        {
            var m = new Matrix(n, n);
            for (int i = 0; i < n; i++)
                m[i, i] = new SimpleComplex(1.0, 0.0);
            m.setDefaultLabels();
            return m;
        }

        public static Matrix Random(int r, int c)
        {
            var m = new Matrix(r, c);
            m.name = "rm";
            var rnd = new Random();
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    m[i, j] = new SimpleComplex(rnd.NextDouble(), 0.0);
            m.setDefaultLabels();
            return m;
        }


        // ---------------------------------------------------------------------
        // Access Property
        // Usage: Matrix m = Matrix (2,3); 
        //        x = m[2,1]
        // ---------------------------------------------------------------------


        // ---------------------------------------------------------------------
        // Simple display method
        // Usage:
        //   m.display();
        // ---------------------------------------------------------------------

        public void display()
        {
            for (int i = 0; i < c; i++)
                Console.Write(columnNames[i] + " ");
            Console.WriteLine();

            for (int i = 0; i < r; i++)
            {
                Console.Write(rowNames[i] + " ");
                for (int j = 0; j < c; j++)
                    Console.Write("{0:F4} ", this[i, j]);
                Console.WriteLine();
            }
        }


        // ---------------------------------------------------------------------
        // Swap rows, i and j
        // ---------------------------------------------------------------------
        public void swapRows(int r1, int r2)
        {
            var tr = new SimpleComplex[c];

            tr = mxx[r1];
            mxx[r1] = mxx[r2];
            mxx[r2] = tr;

            var tmp = (string) rowNames[r1];
            rowNames[r1] = rowNames[r2];
            rowNames[r2] = tmp;
        }


        public static double[][] convertToDouble(Matrix m)
        {
            var result = new double[m.nRows][];
            for (int i = 0; i < m.nRows; i++) result[i] = new double[m.nCols];

            for (int i = 0; i < m.nRows; i++)
                for (int j = 0; j < m.nCols; j++)
                    result[i][j] = m[i, j].Real;
            return result;
        }


        // Add Methods:

        //  Methods that return a new matrix
        //  m = Matrix.add (m1, m2);
        //  m = Matrix.add (m1, 2.3);

        //  Methods that modify the matrix object
        //  m.add (m1);
        //  m.add (2.3); 


        // ---------------------------------------------------------------------
        // Add two matrices together and return the result as a new matrix
        // Usage:
        //   Matrix m3 = Matrix.add (m1, m2);
        // ---------------------------------------------------------------------

        public static Matrix add(Matrix x, Matrix y)
        {
            if (sameDimensions(x, y))
            {
                var result = new Matrix(x.r, x.c);
                for (int i = 0; i <= x.r; i++)
                    for (int j = 0; j <= x.c; j++)
                        result[i, j] = SimpleComplex.complexAdd(x[i, j], y[i, j]);
                return result;
            }
            else
                throw new EMatrixException("Matrices must be the same dimension to perform addition");
        }


        // ---------------------------------------------------------------------
        // Add a matrix to this. this is modified
        // Usage:
        //   m.add (m1)
        // ---------------------------------------------------------------------

        public void add(Matrix x)
        {
            if (sameDimensions(this, x))
            {
                for (int i = 0; i < x.r; i++)
                    for (int j = 0; j < x.c; j++)
                        this[i, j] = SimpleComplex.complexAdd(this[i, j], x[i, j]);
            }
            else
                throw new EMatrixException("Matrices must be the same dimension to perform addition");
        }

        // ---------------------------------------------------------------------
        // Add a scalar value to a matrix and return a new matrix
        // Usage:
        //   Matrix m1 = Matrix.add (m2, Pi);
        // ---------------------------------------------------------------------

        public static Matrix add(Matrix x, double k)
        {
            var result = new Matrix(x.r, x.c);
            for (int i = 0; i < x.r; i++)
                for (int j = 0; j < x.c; j++)
                    result[i, j] = SimpleComplex.complexAdd(x[i, j], new SimpleComplex(k, 0.0));
            return result;
        }


        // ---------------------------------------------------------------------
        // Add a scalar value to a matrix 
        // Usage:
        //   m.add (Pi);
        // ---------------------------------------------------------------------

        public void add(double k)
        {
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    this[i, j] = SimpleComplex.complexAdd(this[i, j], new SimpleComplex(k, 0.0));
        }


        // Sub Methods:

        // Methods that return a new matrix
        // m = Matrix.sub (m1, m2);
        // m = Matrix.sub (m1, 2.3);

        // Methods that modify the matrix object
        // m.sub (m1);
        // m.sub (2.3); 


        // ---------------------------------------------------------------------
        // Subtract two matrices together and return the result as a new matrix
        // Usage:
        //   Matrix m3 = Matrix.sub (m1, m2);
        // ---------------------------------------------------------------------

        public static Matrix sub(Matrix x, Matrix y)
        {
            if (sameDimensions(x, y))
            {
                var result = new Matrix(x.r, x.c);
                for (int i = 0; i < x.r; i++)
                    for (int j = 0; j < x.c; j++)
                        result[i, j] = SimpleComplex.complexSub(x[i, j], y[i, j]);
                return result;
            }
            else
                throw new EMatrixException("Matrices must be the same dimension to perform addition");
        }


        // ---------------------------------------------------------------------
        // Subtract a matrix from this. this is modified
        // Usage:
        //   m.sub (m1);
        // ---------------------------------------------------------------------

        public void sub(Matrix x)
        {
            if (sameDimensions(this, x))
            {
                for (int i = 0; i < x.r; i++)
                    for (int j = 0; j < x.c; j++)
                        this[i, j] = SimpleComplex.complexSub(this[i, j], x[i, j]);
            }
            else
                throw new EMatrixException("Matrices must be the same dimension to perform subtraction");
        }


        // ---------------------------------------------------------------------
        // Subtract a scalar value from a matrix and return a new matrix
        // Usage:
        //   Matrix m1 = Matrix.sub (m2, 2.718);
        // ---------------------------------------------------------------------

        public static Matrix sub(Matrix x, double k)
        {
            var result = new Matrix(x.r, x.c);
            for (int i = 0; i < x.r; i++)
                for (int j = 0; j < x.c; j++)
                    result[i, j] = SimpleComplex.complexSub(x[i, j], new SimpleComplex(k, 0.0));
            return result;
        }


        // ---------------------------------------------------------------------
        // Subtract a scalar value from a matrix
        // Usage:
        //   m.sub (3.1415);
        // ---------------------------------------------------------------------

        public void sub(double k)
        {
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    this[i, j] = SimpleComplex.complexSub(this[i, j], new SimpleComplex(k, 0.0));
        }


        // ********************************************************************
        // Multiply matrix 'm1' by scalar k to give result in Self                  
        //                                                                         
        // Usage:  A.mult (A1, A2); multiply A1 by A2 giving A                  
        //                                                                      
        // ******************************************************************** 
        public Matrix mult(Matrix m, double k)
        {
            for (int i = 0; i < m.r; i++)
                for (int j = 0; j < m.c; j++)
                    this[i, j] = SimpleComplex.complexMult(m[i, j], new SimpleComplex(k, 0.0));
            return this;
        }


        // ********************************************************************
        // Multiply matrix 'm1' by complex number z to give result in this                  
        //                                                                         
        // Usage:  A.mult (A1, z); multiply A1 by z giving A                  
        //                                                                      
        // ******************************************************************** 
        public Matrix mult(Matrix m, SimpleComplex z)
        {
            for (int i = 0; i < m.r; i++)
                for (int j = 0; j < m.c; j++)
                    this[i, j] = SimpleComplex.complexMult(m[i, j], z);
            return this;
        }

        // ********************************************************************
        // Multiply the diagonal of the matrix 'm1' by complex number z to give 
        // result in this                  
        //                                                                         
        // Usage:  A.multDiag (A1, A2); multiply A1 by A2 giving A                  
        //                                                                      
        // ******************************************************************** 
        public Matrix multDiag(Matrix m, SimpleComplex z)
        {
            for (int i = 0; i < m.r; i++)
                this[i, i] = SimpleComplex.complexMult(m[i, i], z);
            return this;
        }

        // ********************************************************************
        // Multiply matrix 'm1' by 'm2' to give result in this                  
        //                                                                         
        // Usage:  A.mult (A1, A2); multiply A1 by A2 giving A                  
        //                                                                      
        // ******************************************************************** 
        public Matrix mult(Matrix m1, Matrix m2)
        {
            if (m1.c == m2.r)
            {
                resize(m1.r, m2.c);
                int m1_col = m1.c;
                for (int i = 0; i < r; i++)
                    for (int j = 0; j < m2.c; j++)
                    {
                        var sum = new SimpleComplex(0.0, 0.0);
                        for (int k = 0; k < m1_col; k++)
                            sum = SimpleComplex.complexAdd(sum, SimpleComplex.complexMult(m1[i, k], m2[k, j]));
                        this[i, j] = sum;
                    }
                return this;
            }
            else
                throw new EMatrixSizeError("Incompatible matrix operands to multiply");
        }

        // ---------------------------------------------------------------------
        // Determine the transpose of a matrix
        // Usage:
        //   m.tranpose();
        // ---------------------------------------------------------------------

        public void transpose()
        {
            //Self.FRowNames.clear;
            //Self.FColumnNames.clear;
            //if m.c > 0 then Self.FRowNames.Assign (m.FColumnNames);
            //if m.r > 0 then Self.FColumnNames.Assign (m.FRowNames);
            //result := Self;

            int ir = r, ic = c;
            var m = new Matrix(ic, ir);
            for (int i = 0; i < ir; i++)
                for (int j = 0; j < ic; j++)
                    m[j, i] = this[i, j];
            mxx = m.mxx;
            r = ic;
            c = ir;
            ArrayList tmp = rowNames;
            rowNames = columnNames;
            columnNames = tmp;
        }


        // Reduce to complete row echelon form
        // By James D. Reilly, Fort Wayne, Indiana
        /*public double rowEchelon(ref int rank) {
            int x;
            double Pivot, Temp;
            double Determinant;
            int PivotRow, PivotCol;

            PivotRow = 0;
            PivotCol = 0;
            Determinant = 1.0;
            do {
                // Find largest pivot.  Search for a number below
                // the current pivot with a greater absolute value.
                x = PivotRow;
                for(int C=PivotRow; C<nRows; C++)
                    if(Math.Abs(this[C, PivotCol]) > Math.Abs(this[x, PivotCol]))
                        x = C;

                if(x != PivotRow) {
                    // If here, there is a better pivot choice somewhere
                    // below the current pivot.
                    // Interchange the pivot row with the row
                    // containing the largest pivot, x
                    for(int b=0; b<nCols; b++) {
                        Temp = this[PivotRow, b];
                        this[PivotRow, b] = this[x, b];
                        this[x, b] = Temp;
                    }

                    Determinant = -Determinant;
                }

                Pivot = this[PivotRow, PivotCol];
                Determinant = Pivot*Determinant;

                if(Pivot != 0.0) {
                    // Introduce a '1' at the pivot point
                    for(int b=0; b < nCols; b++)
                        this[PivotRow, b] = this[PivotRow, b]/Pivot;
						     
                    for(int b=0; b < nRows; b++) {
                        // Eliminate (make zero) all elements above
                        // and below the pivot.
                        // Skip over the pivot row when we come to it.
                        if(b != PivotRow) {
                            Pivot = this[b, PivotCol];
                            for(int C = PivotRow; C < nCols; C++)
                                this[b, C] = this[b, C] - this[PivotRow, C]*Pivot;
                        }
                    }
                    PivotRow++; // Next row
                }
                PivotCol++;  // Next column
            }
            while((PivotRow < nRows) && (PivotCol < nCols)); // Reached an edge yet?

            // Finally compute the rank
			double[] sq = new double[this.r];
			for (int i = 0; i < this.r; i++)
				for (int j = 0; j < this.c; j++)
					sq[i] = sq[i] + Math.Abs (this[i,j]);
			rank = 0;
			for (int i = 0; i < this.r; i++)
                if (sq[i] < 1E-6)
					rank++;
			return Determinant;
        }*/
    }
}