//-*-C++-*-

#ifndef UTILS_BLAS_GEMM_H
#define UTILS_BLAS_GEMM_H

namespace blas
{
  /*
  extern "C" void sgemv_(const char* trans,
			 const int* M, const int* N, const float *alpha, const float* A , const int* LDA,
			 const float* x, const int *incx,
			 const float* beta,
			 float* y, const int *incy);
  */
  
  extern "C" void sgemm_(const char* transA, const char* transB,
			 const int* M, const int* N, const int* K,
			 const float *alpha,
			 const float* A, const int* LDA,
			 const float* B, const int *LDB,
			 const float* beta,
			 const float* C, const int *LDC);

  template<typename value_type>
  struct gemm
  {};

  template<>
  struct gemm<float>
  {
    static void execute(const char transA, const char transB,
		 const int M, const int N, const int K,
		 const float alpha,
		 const float* A, const int LDA,
		 const float* B, const int LDB,
		 const float beta,
		 const float* C, const int LDC)
    {
      sgemm_(&transA, &transB, &M, &N, &K, &alpha, A, &LDA, B, &LDB, &beta, C, &LDC);
    }
  };  
}

#endif
