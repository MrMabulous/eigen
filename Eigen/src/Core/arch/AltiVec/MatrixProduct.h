// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2020 Everton Constantino (everton.constantino@ibm.com)
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_MATRIX_PRODUCT_ALTIVEC_H
#define EIGEN_MATRIX_PRODUCT_ALTIVEC_H

#ifdef __MMA__

namespace Eigen {

namespace internal {

const int accRows = 4;
const int accCols = 4;
const int floatVectorSize = 4;

typedef struct
{
  __vector float v0;
  __vector float v1;
  __vector float v2;
  __vector float v3;
} struct_v4sf_t;

union RESULT
{
  __struct_quad sc;
  struct_v4sf_t sf;
};

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
struct gemm_pack_lhs<float, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode>
{
  void operator()(float* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, bool Conjugate, bool PanelMode>
void gemm_pack_lhs<float, Index, DataMapper, Pack1, Pack2, Packet, ColMajor, Conjugate, PanelMode>
  ::operator()(float* blockA, const DataMapper& lhs, Index depth, Index rows, Index stride, Index offset)
{
    int ri = 0, j;
    for(j = 0; j + floatVectorSize < rows; j+=floatVectorSize)
    {
        for(int i = 0; i < depth; i++)
        {
            blockA[ri+0] = lhs(j+0,i);
            blockA[ri+1] = lhs(j+1,i);
            blockA[ri+2] = lhs(j+2,i);
            blockA[ri+3] = lhs(j+3,i);
            ri += floatVectorSize;
        }     
    }
    for(int i = 0; i < depth; i++)
    {
        int k = j;
        for(; k < rows; k++)
        {
            blockA[ri] = lhs(k, i);
            ri += 1;
        }
    }
}

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
struct gemm_pack_rhs<float, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>
{
  void operator()(float* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride=0, Index offset=0);
};

template<typename Index, typename DataMapper, int nr, bool Conjugate, bool PanelMode>
void gemm_pack_rhs<float, Index, DataMapper, nr, ColMajor, Conjugate, PanelMode>
  ::operator()(float* blockB, const DataMapper& rhs, Index depth, Index cols, Index stride, Index offset)
{
    int ri = 0, j;
    for(j = 0; j + floatVectorSize < cols; j+=floatVectorSize)
    {
        for(int i = 0; i < depth; i++)
        {
            blockB[ri+0] = rhs(i, j+0);
            blockB[ri+1] = rhs(i, j+1);
            blockB[ri+2] = rhs(i, j+2);
            blockB[ri+3] = rhs(i, j+3);
            ri += floatVectorSize;
        }     
    }
    for(int i = 0; i < depth; i++)
    {
        int k = j;
        for(; k < cols; k++)
        {
            blockB[ri] = rhs(i, k);
            ri += 1;
        }
    }
}

template<typename RhsScalar, typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
struct gebp_kernel<float, RhsScalar, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
{
  typedef typename DataMapper::LinearMapper LinearMapper;

  void operator()(const DataMapper& res, const float* blockA, const RhsScalar* blockB,
                  Index rows, Index depth, Index cols, float alpha,
                  Index strideA=-1, Index strideB=-1, Index offsetA=0, Index offsetB=0);
};

template<typename RhsScalar, typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs, bool ConjugateRhs>
void gebp_kernel<float, RhsScalar, Index, DataMapper, mr, nr, ConjugateLhs, ConjugateRhs>
  ::operator()(const DataMapper& res, const float* blockA, const RhsScalar* blockB,
               Index rows, Index depth, Index cols, float alpha,
               Index strideA, Index strideB, Index offsetA, Index offsetB)
  {
      const int remaining_rows = rows % accRows;
      const int remaining_cols = cols % accCols;
      const int remaining_depth = depth % floatVectorSize;
      const int timesRows = (rows / accRows);
      const int timesCols = (cols / accCols);

      int row;
      for(row = 0; row + accRows <= rows; row += accRows)
      {
          const float *rhs_base = blockB;
          const float *lhs_base = blockA + (row/accRows)*depth*floatVectorSize;

          int col;
          for(col = 0; col + accCols <= cols; col += accCols){
              const float *rhs_ptr = rhs_base + (col/accCols)*depth*floatVectorSize;
              const float *lhs_ptr = lhs_base;
             
              __vector_quad acc = __builtin_mma_xxsetaccz();
              for(int k = 0; k < depth; k++)
              {
                  __vector float lhsV = *((__vector float *)lhs_ptr);
                  __vector float rhsV = *((__vector float *)rhs_ptr);

                  acc = __builtin_mma_xvf32gerpp(acc, (__vector unsigned char) lhsV, (__vector unsigned char) rhsV);
                 
                  lhs_ptr += floatVectorSize;
                  rhs_ptr += floatVectorSize;
              }
              RESULT result;
              result.sc  =  __builtin_mma_disassemble_acc(acc);
            
              res.scatterPacket(row    , col, result.sf.v3);
              res.scatterPacket(row + 1, col, result.sf.v2);
              res.scatterPacket(row + 2, col, result.sf.v1);
              res.scatterPacket(row + 3, col, result.sf.v0);              
          }
         
          if(remaining_cols > 0)
          {
              const float *rhs_ptr = rhs_base + (col/accCols)*depth*floatVectorSize;
              const float *lhs_ptr = lhs_base;
              for(int k = 0; k < depth; k++)
              {
                 for(int arow = 0; arow < accRows; arow++)
                 {
                     for(int acol = 0; acol < remaining_cols; acol++ )
                     {
                        res(row + arow, col + acol) += lhs_ptr[arow]*rhs_ptr[acol];
                     }
                 }
                 rhs_ptr += remaining_cols;
                 lhs_ptr += floatVectorSize;
              }
          }
      }

      if(remaining_rows > 0)
      {
          const float *rhs_base = blockB;
          const float *lhs_base = blockA + (row/accRows)*depth*floatVectorSize;

          int col;
          for(col = 0; col + accCols <= cols; col += accCols)
          {
              const float *rhs_ptr = rhs_base + (col/accCols)*depth*floatVectorSize;
              const float *lhs_ptr = lhs_base;
              for(int k = 0; k < depth; k++)
              {
                 for(int arow = 0; arow < remaining_rows; arow++)
                 {
                     for(int acol = 0; acol < accCols; acol++ )
                     {
                        res(row + arow, col + acol) += lhs_ptr[arow]*rhs_ptr[acol];
                     }
                 }
                 rhs_ptr += floatVectorSize;
                 lhs_ptr += remaining_rows;
              }
          }
         
          if(remaining_cols > 0)
          {
              const float *rhs_ptr = rhs_base + (col/accCols)*depth*floatVectorSize;
              const float *lhs_ptr = lhs_base;
              for(int k = 0; k < depth; k++)
              {
                 for(int arow = 0; arow < remaining_rows; arow++)
                 {
                     for(int acol = 0; acol < remaining_cols; acol++ )
                     {
                        res(row + arow, col + acol) += lhs_ptr[arow]*rhs_ptr[acol];
                     }
                 }
                 rhs_ptr += remaining_cols;
                 lhs_ptr += remaining_rows;
              }
          }
      }
  }

} // end namespace internal

} // end namespace Eigen

#endif // __MMA__

#endif // EIGEN_MATRIX_PRODUCT_ALTIVEC_H
