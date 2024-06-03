//-*-C++-*-

#ifndef UTILS_LLM_OP_MATMUL_H
#define UTILS_LLM_OP_MATMUL_H

namespace llmcpp
{
  template<typename index_type, typename value_type>
  class matmul
  {

  public:

    static void forward(float* out,
                        float* inp, float* weight, float* bias,
                        int B, int T, int C, int OC);

    static void backward(float* dinp, float* dweight, float* dbias,
                         float* dout, float* inp, float* weight,
                         int B, int T, int C, int OC);

    static bool test_forward();
    static bool test_backward();

  private:

    static void forward_orig(float* out,
                             float* inp, float* weight, float* bias,
                             int B, int T, int C, int OC);

    static void forward_blas(float* out,
                             float* inp, float* weight, float* bias,
                             int B, int T, int C, int OC);

    static void backward_orig(float* dinp, float* dweight, float* dbias,
                              float* dout, float* inp, float* weight,
                              int B, int T, int C, int OC);

    static void backward_blas(float* dinp, float* dweight, float* dbias,
                              float* dout, float* inp, float* weight,
                              int B, int T, int C, int OC);
  };

  template<typename index_type, typename value_type>
  void matmul<index_type, value_type>::forward(float* out,
                                               float* inp, float* weight, float* bias,
                                               int B, int T, int C, int OC)
  {
    forward_blas(out, inp, weight, bias, B, T, C, OC);
  }

  template<typename index_type, typename value_type>
  void matmul<index_type, value_type>::forward_orig(float* out,
                                                    float* inp, float* weight, float* bias,
                                                    int B, int T, int C, int OC) {
    //LOG_S(INFO) << "matmul::" << __FUNCTION__;

    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    //#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
        float* out_bt = out + b * T * OC + t * OC;
        float* inp_bt = inp + b * T * C + t * C;
        for (int o = 0; o < OC; o++) {
          float val = (bias != NULL) ? bias[o] : 0.0f;
          float* wrow = weight + o*C;
          for (int i = 0; i < C; i++) {
            val += inp_bt[i] * wrow[i];
          }
          out_bt[o] = val;
        }
      }
    }
  }

  template<typename index_type, typename value_type>
  void matmul<index_type, value_type>::forward_blas(float* out,
                                                    float* inp, float* weight, float* bias,
                                                    int B, int T, int C, int OC) {
    //LOG_S(INFO) << "matmul::" << __FUNCTION__;

    // most of the running time is spent here and in matmul_backward
    // OC is short for "output channels"
    // inp is (B,T,C), weight is (OC, C), bias is (OC)
    // out will be (B,T,OC)

    float alpha = 1.0, beta = 0.0;
    if(bias != NULL)
      {
        beta = 1.0;

        float* out_bt = NULL;
        for (int b = 0; b < B; b++) {
          for (int t = 0; t < T; t++) {

            out_bt = out + b * T * OC + t * OC;
            for (int o = 0; o < OC; o++) {
              out_bt[o] = bias[o];
            }
          }
        }
      }

    // row-major = inp is (B,T,C), weight is (OC, C), bias is (OC), out is (B,T,OC)
    // col-major = inp is (C,T,B), weight is (C, OC), bias is (OC), out is (OC,T,B)
    // out = weight^T x inp

    blas::gemm<value_type>::execute('T', 'N', OC, B*T, C, alpha,
                                    weight, C,
                                    inp, C,
                                    beta, out, OC);
  }

  template<typename index_type, typename value_type>
  bool matmul<index_type, value_type>::test_forward()
  {
    int B=3, T=5, C=7, OC=13;

    dense_tensor<int, float> A, W, C1, C2;
    A.initialise("A", {B, T, C}, false).to_rand();
    W.initialise("W", {OC, C}, false).to_rand();

    C1.initialise("C1", {B, T, OC}, false).to_zero();
    C2.initialise("C2", {B, T, OC}, false).to_zero();

    forward_orig(C1.ptr(), A.ptr(), W.ptr(), NULL, B, T, C, OC);
    forward_blas(C2.ptr(), A.ptr(), W.ptr(), NULL, B, T, C, OC);

    value_type maxdiff = C1.max_diff(C2);
    LOG_S(INFO) << "diff: " << maxdiff;

    return true;
  }

  template<typename index_type, typename value_type>
  bool matmul<index_type, value_type>::test_backward()
  {
    int B=3, T=5, C=7, OC=13;

    dense_tensor<int, float> inp, dinp1, dinp2, weight, dweight1, dweight2, dout;

    inp.initialise("inp", {B, T, C}, false).to_rand();
    weight.initialise("weight", {OC, C}, false).to_rand();
    dout.initialise("dout", {B, T, OC}, false).to_rand();

    dinp1.initialise("dinp1", {B, T, C}, false).to_zero();
    dinp2.initialise("dinp2", {B, T, C}, false).to_zero();

    dweight1.initialise("dweight1", {OC, C}, false).to_zero();
    dweight2.initialise("dweight2", {OC, C}, false).to_zero();

    backward_orig(dinp1.ptr(), dweight1.ptr(), NULL,
                  dout.ptr(), inp.ptr(), weight.ptr(),
                  B, T, C, OC);

    backward_blas(dinp2.ptr(), dweight2.ptr(), NULL,
                  dout.ptr(), inp.ptr(), weight.ptr(),
                  B, T, C, OC);

    value_type max_diff = 0;

    max_diff = dinp1.max_diff(dinp2);
    LOG_S(INFO) << "diff dinp: " << max_diff;

    max_diff = dweight1.max_diff(dweight2);
    LOG_S(INFO) << "diff dweight: " << max_diff;

    return true;
  }

  template<typename index_type, typename value_type>
  void matmul<index_type, value_type>::backward(float* dinp, float* dweight, float* dbias,
                                                float* dout, float* inp, float* weight,
                                                int B, int T, int C, int OC) {
    backward_blas(dinp, dweight, dbias,
                  dout, inp, weight,
                  B, T, C, OC);
  }

  template<typename index_type, typename value_type>
  void matmul<index_type, value_type>::backward_orig(float* dinp, float* dweight, float* dbias,
                                                     float* dout, float* inp, float* weight,
                                                     int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T

    //#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
        float* dout_bt = dout + b * T * OC + t * OC;
        float* dinp_bt = dinp + b * T * C + t * C;
        for (int o = 0; o < OC; o++) {
          float* wrow = weight + o*C;
          float d = dout_bt[o];
          for (int i = 0; i < C; i++) {
            dinp_bt[i] += wrow[i] * d;
          }
        }
      }
    }

    // backward into weight/bias, parallelize over output channels OC

    //#pragma omp parallel for
    for (int o = 0; o < OC; o++) {
      for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
          float* dout_bt = dout + b * T * OC + t * OC;
          float* inp_bt = inp + b * T * C + t * C;
          float* dwrow = dweight + o*C;
          float d = dout_bt[o];
          if (dbias != NULL) { dbias[o] += d; }
          for (int i = 0; i < C; i++) {
            dwrow[i] += inp_bt[i] * d;
          }
        }
      }
    }
  }

  template<typename index_type, typename value_type>
  void matmul<index_type, value_type>::backward_blas(float* dinp, float* dweight, float* dbias,
                                                     float* dout, float* inp, float* weight,
                                                     int B, int T, int C, int OC) {
    // most of the running time is spent here and in matmul_forward
    // this backward could be done in a single "round" of loops
    // but that doesn't afford an efficient parallelization strategy

    // backward into inp first, parallelize over B,T

    // row-major = dinp is (B,T,C), weight is (OC, C), dout is (B,T,OC)
    // col-major = dinp is (C,T,B), weight is (C, OC), dout is (OC,T,B)
    //  => dinp = weight x dout
    {
      value_type alpha = 1.0, beta = 0.0;
      blas::gemm<value_type>::execute('N', 'N', C, B*T, OC, alpha,
                                      weight, C,
                                      dout, OC,
                                      beta, dinp, C);
    }
    // backward into bias, parallelize over output channels OC

    if(dbias != NULL)
      {
        for (int o = 0; o < OC; o++) {
          for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
              float* dout_bt = dout + b * T * OC + t * OC;
              float d = dout_bt[o];
              dbias[o] += d;
            }
          }
        }
      }

    // backward into weight, parallelize over output channels OC

    // row-major = inp is (B,T,C), dweight is (OC, C), dout is (B,T,OC)
    // col-major = inp is (C,T,B), dweight is (C, OC), dout is (OC,T,B)
    //  => dweight = inp x dout^T

    {
      value_type alpha = 1.0, beta = 0.0;
      blas::gemm<value_type>::execute('N', 'T', C, OC, B*T, alpha,
                                      inp, C,
                                      dout, OC,
                                      beta, dweight, C);
    }
  }

}

#endif
