//-*-C++-*-

#ifndef LLM_OP_H
#define LLM_OP_H

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

namespace llmcpp
{
  template<typename index_type, typename value_type>
  class llm_base_op
  {


  };


  template<typename index_type, typename value_type>
  class encoder
  {

  public:

    static void forward(float* out,
                        int* inp, float* wte, float* wpe,
                        int B, int T, int C);

    static void backward(float* dwte, float* dwpe,
                         float* dout, int* inp,
                         int B, int T, int C);
  };

  template<typename index_type, typename value_type>
  void encoder<index_type, value_type>::forward(float* out,
                                                int* inp, float* wte, float* wpe,
                                                int B, int T, int C) {
    //LOG_S(INFO) << "encoder::" << __FUNCTION__;
    
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
        // seek to the output position in out[b,t,:]
        float* out_bt = out + b * T * C + t * C;
        // get the index of the token at inp[b, t]
        int ix = inp[b * T + t];
        // seek to the position in wte corresponding to the token
        float* wte_ix = wte + ix * C;
        // seek to the position in wpe corresponding to the position
        float* wpe_t = wpe + t * C;
        // add the two vectors and store the result in out[b,t,:]
        for (int i = 0; i < C; i++) {
          out_bt[i] = wte_ix[i] + wpe_t[i];
        }
      }
    }
  }

  template<typename index_type, typename value_type>
  void encoder<index_type, value_type>::backward(float* dwte, float* dwpe,
                                                 float* dout, int* inp,
                                                 int B, int T, int C) {
    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
        float* dout_bt = dout + b * T * C + t * C;
        int ix = inp[b * T + t];
        float* dwte_ix = dwte + ix * C;
        float* dwpe_t = dwpe + t * C;
        for (int i = 0; i < C; i++) {
          float d = dout_bt[i];
          dwte_ix[i] += d;
          dwpe_t[i] += d;
        }
      }
    }
  }

  template<typename index_type, typename value_type>
  class layernorm
  {
  public:

    static void forward(float* out, float* mean, float* rstd,
                 float* inp, float* weight, float* bias,
                 int B, int T, int C);

    static void backward(float* dinp, float* dweight, float* dbias,
                  float* dout, float* inp, float* weight, float* mean, float* rstd,
                  int B, int T, int C);
  };

  template<typename index_type, typename value_type>
  void layernorm<index_type, value_type>::forward(float* out, float* mean, float* rstd,
                                                  float* inp, float* weight, float* bias,
                                                  int B, int T, int C) {
    //LOG_S(INFO) << "layernorm::" << __FUNCTION__;
    
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
    // both inp and out are (B,T,C) of the activations
    // mean and rstd are (B,T) buffers, to be used later in backward pass
    // at each position (b,t) of the input, the C-dimensional vector
    // of activations gets normalized, then scaled and shifted
    float eps = 1e-5f;
    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
        // seek to the input position inp[b,t,:]
        float* x = inp + b * T * C + t * C;
        // calculate the mean
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
          m += x[i];
        }
        m = m/C;
        // calculate the variance (without any bias correction)
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
          float xshift = x[i] - m;
          v += xshift * xshift;
        }
        v = v/C;
        // calculate the rstd (reciprocal standard deviation)
        float s = 1.0f / sqrtf(v + eps);
        // seek to the output position in out[b,t,:]
        float* out_bt = out + b * T * C + t * C;
        for (int i = 0; i < C; i++) {
          float n = (s * (x[i] - m)); // normalize
          float o = n * weight[i] + bias[i]; // scale and shift
          out_bt[i] = o; // write
        }
        // cache the mean and rstd for the backward pass later
        mean[b * T + t] = m;
        rstd[b * T + t] = s;
      }
    }
  }

  template<typename index_type, typename value_type>
  void layernorm<index_type, value_type>::backward(float* dinp, float* dweight, float* dbias,
                                                   float* dout, float* inp, float* weight, float* mean, float* rstd,
                                                   int B, int T, int C) {
    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
        float* dout_bt = dout + b * T * C + t * C;
        float* inp_bt = inp + b * T * C + t * C;
        float* dinp_bt = dinp + b * T * C + t * C;
        float mean_bt = mean[b * T + t];
        float rstd_bt = rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = 0; i < C; i++) {
          float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
          float dnorm_i = weight[i] * dout_bt[i];
          dnorm_mean += dnorm_i;
          dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // now iterate again and accumulate all the gradients
        for (int i = 0; i < C; i++) {
          float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
          float dnorm_i = weight[i] * dout_bt[i];
          // gradient contribution to bias
          dbias[i] += dout_bt[i];
          // gradient contribution to weight
          dweight[i] += norm_bti * dout_bt[i];
          // gradient contribution to input
          float dval = 0.0f;
          dval += dnorm_i; // term 1
          dval -= dnorm_mean; // term 2
          dval -= norm_bti * dnorm_norm_mean; // term 3
          dval *= rstd_bt; // final scale
          dinp_bt[i] += dval;
        }
      }
    }
  }


  
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

    static bool forward_test();
    
  private:

    static void forward_orig(float* out,
			     float* inp, float* weight, float* bias,
			     int B, int T, int C, int OC);

    static void forward_blas(float* out,
			     float* inp, float* weight, float* bias,
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
    
    blas::gemm<value_type>::execute('T', 'N', OC, B*T, C, alpha,
				    weight, C, 
				    inp, C,
				    beta, out, OC);
  }

  template<typename index_type, typename value_type>
  bool matmul<index_type, value_type>::forward_test()
  {
    int B=3, T=5, C=7, OC=13;

    llm_tensor<int, float> A, W, C1, C2;
    A.initialise("A", {B, T, C}, false);
    W.initialise("W", {C, OC}, false);
    
    C1.initialise("C1", {B, T, OC}, false);
    C2.initialise("C2", {B, T, OC}, false);

    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
	for (int i = 0; i < C; i++) {
	  A(b, t, i) = b + t + i;
	}
      }
    }
    
    for (int i = 0; i < C; i++) {
      for (int o = 0; o < OC; o++) {
	W(i, o) = (o-4) * i;
      }
    }
    
    forward_orig(C1.ptr(), A.ptr(), W.ptr(), NULL, B, T, C, OC);
    forward_blas(C2.ptr(), A.ptr(), W.ptr(), NULL, B, T, C, OC);

    value_type maxdiff=0;
    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
	for (int i = 0; i < OC; i++) {
	  value_type diff = std::abs(C1(b,t,i)-C2(b,t,i));
	  maxdiff = maxdiff>diff? maxdiff : diff; 
	}
      }
    }

    LOG_S(INFO) << "diff: " << maxdiff;
    return true;
  }
  
  template<typename index_type, typename value_type>
  void matmul<index_type, value_type>::backward(float* dinp, float* dweight, float* dbias,
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
  class attention
  {
  public:

    static void forward(float* out, float* preatt, float* att,
                        float* inp,
                        int B, int T, int C, int NH);

    static void backward(float* dinp, float* dpreatt, float* datt,
                         float* dout, float* inp, float* att,
                         int B, int T, int C, int NH);
  };

  template<typename index_type, typename value_type>
  void attention<index_type, value_type>::forward(float* out, float* preatt, float* att,
                                                  float* inp,
                                                  int B, int T, int C, int NH) {
    //LOG_S(INFO) << "attantion::" << __FUNCTION__;

    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);
    
    //#pragma omp parallel for collapse(3)
    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
        for (int h = 0; h < NH; h++) {
          float* query_t = inp + b * T * C3 + t * C3 + h * hs;
          float* preatt_bth = preatt + b*NH*T*T + h*T*T + t*T;
          float* att_bth = att + b*NH*T*T + h*T*T + t*T;

          // pass 1: calculate query dot key and maxval
          float maxval = -10000.0f; // TODO something better
          for (int t2 = 0; t2 <= t; t2++) {
            float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

            // (query_t) dot (key_t2)
            float val = 0.0f;
            for (int i = 0; i < hs; i++) {
              val += query_t[i] * key_t2[i];
            }
            val *= scale;
            if (val > maxval) {
              maxval = val;
            }

            preatt_bth[t2] = val;
          }

          // pass 2: calculate the exp and keep track of sum
          // maxval is being calculated and subtracted only for numerical stability
          float expsum = 0.0f;
          for (int t2 = 0; t2 <= t; t2++) {
            float expv = expf(preatt_bth[t2] - maxval);
            expsum += expv;
            att_bth[t2] = expv;
          }
          float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

          // pass 3: normalize to get the softmax
          for (int t2 = 0; t2 < T; t2++) {
            if (t2 <= t) {
              att_bth[t2] *= expsum_inv;
            } else {
              // causal attention mask. not strictly necessary to set to zero here
              // only doing this explicitly for debugging and checking to PyTorch
              att_bth[t2] = 0.0f;
            }
          }

          // pass 4: accumulate weighted values into the output of attention
          float* out_bth = out + b * T * C + t * C + h * hs;
          for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
          for (int t2 = 0; t2 <= t; t2++) {
            float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
            float att_btht2 = att_bth[t2];
            for (int i = 0; i < hs; i++) {
              out_bth[i] += att_btht2 * value_t2[i];
            }
          }
        }
      }
    }
  }

  template<typename index_type, typename value_type>
  void attention<index_type, value_type>::backward(float* dinp, float* dpreatt, float* datt,
                                                   float* dout, float* inp, float* att,
                                                   int B, int T, int C, int NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    int C3 = C*3;
    int hs = C / NH; // head size
    float scale = 1.0 / sqrtf(hs);

    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
        for (int h = 0; h < NH; h++) {
          float* att_bth = att + b*NH*T*T + h*T*T + t*T;
          float* datt_bth = datt + b*NH*T*T + h*T*T + t*T;
          float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t*T;
          float* dquery_t = dinp + b * T * C3 + t * C3 + h * hs;
          float* query_t = inp + b * T * C3 + t * C3 + h * hs;

          // backward pass 4, through the value accumulation
          float* dout_bth = dout + b * T * C + t * C + h * hs;
          for (int t2 = 0; t2 <= t; t2++) {
            float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
            float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
            for (int i = 0; i < hs; i++) {
              // in the forward pass this was:
              // out_bth[i] += att_bth[t2] * value_t2[i];
              // so now we have:
              datt_bth[t2] += value_t2[i] * dout_bth[i];
              dvalue_t2[i] += att_bth[t2] * dout_bth[i];
            }
          }

          // backward pass 2 & 3, the softmax
          // note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
          for (int t2 = 0; t2 <= t; t2++) {
            for (int t3 = 0; t3 <= t; t3++) {
              float indicator = t2 == t3 ? 1.0f : 0.0f;
              float local_derivative = att_bth[t2] * (indicator - att_bth[t3]);
              dpreatt_bth[t3] += local_derivative * datt_bth[t2];
            }
          }

          // backward pass 1, the query @ key matmul
          for (int t2 = 0; t2 <= t; t2++) {
            float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
            float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key
            for (int i = 0; i < hs; i++) {
              // in the forward pass this was:
              // preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
              // so now we have:
              dquery_t[i] += key_t2[i] * dpreatt_bth[t2] * scale;
              dkey_t2[i] += query_t[i] * dpreatt_bth[t2] * scale;
            }
          }
        }
      }
    }
  }


  template<typename index_type, typename value_type>
  class gelu
  {
    const static inline value_type GELU_SCALING_FACTOR=std::sqrt(2.0f / M_PI);

  public:

    static void forward(float* out, float* inp, int N);
    static void backward(float* dinp, float* inp, float* dout, int N);
  };

  template<typename index_type, typename value_type>
  void gelu<index_type, value_type>::forward(float* out, float* inp, int N) {
    //LOG_S(INFO) << "gelu::" << __FUNCTION__;

    // (approximate) GeLU elementwise non-linearity in the MLP block of Transformer
    for (int i = 0; i < N; i++) {
      float x = inp[i];
      float cube = 0.044715f * x * x * x;
      out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
    }
  }

  // we want to use -Ofast optimization, but sadly GeLU breaks, so disable this flag just for it (#168)
  //#pragma float_control(precise, on, push)
  //#if defined(__GNUC__) && !defined(__clang__)
  //__attribute__((optimize("no-finite-math-only")))
  //#endif
  template<typename index_type, typename value_type>
  void gelu<index_type, value_type>::backward(float* dinp, float* inp, float* dout, int N) {
    for (int i = 0; i < N; i++) {
      float x = inp[i];
      float cube = 0.044715f * x * x * x;
      float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
      float tanh_out = tanhf(tanh_arg);
      float coshf_out = coshf(tanh_arg);
      float sech_out = 1.0f / (coshf_out * coshf_out);
      float local_grad = 0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR * (1.0f + 3.0f * 0.044715f * x * x);
      dinp[i] += local_grad * dout[i];
    }
  }
  //#pragma float_control(pop)

  template<typename index_type, typename value_type>
  class residual
  {
  public:

    static void forward(float* out, float* inp1, float* inp2, int N);
    static void backward(float* dinp1, float* dinp2, float* dout, int N);
  };

  template<typename index_type, typename value_type>
  void residual<index_type, value_type>::forward(float* out, float* inp1, float* inp2, int N) {
    //LOG_S(INFO) << "residual::" << __FUNCTION__;
    
    for (int i = 0; i < N; i++) {
      out[i] = inp1[i] + inp2[i];
    }
  }

  template<typename index_type, typename value_type>
  void residual<index_type, value_type>::backward(float* dinp1, float* dinp2, float* dout, int N) {
    for (int i = 0; i < N; i++) {
      dinp1[i] += dout[i];
      dinp2[i] += dout[i];
    }
  }


  template<typename index_type, typename value_type>
  class softmax
  {
  public:
    
    static void forward(float* probs, float* logits, int B, int T, int V, int Vp);
  };

  template<typename index_type, typename value_type>
  void softmax<index_type, value_type>::forward(float* probs, float* logits, int B, int T, int V, int Vp) {
    //LOG_S(INFO) << "softmax::" << __FUNCTION__;

    // output: probs are (B,T,Vp) of the probabilities (sums to 1.0 in each b,t position)
    // input: logits is (B,T,Vp) of the unnormalized log probabilities
    // Vp is the padded vocab size (for efficiency), V is the "real" vocab size
    // example: Vp is 50304 and V is 50257

    //#pragma omp parallel for collapse(2)
    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
        // probs <- softmax(logits)
        float* logits_bt = logits + b * T * Vp + t * Vp;
        float* probs_bt = probs + b * T * Vp + t * Vp;

        // maxval is only calculated and subtracted for numerical stability
        float maxval = -10000.0f; // TODO something better
        for (int i = 0; i < V; i++) {
          if (logits_bt[i] > maxval) {
            maxval = logits_bt[i];
          }
        }
        float sum = 0.0f;
        for (int i = 0; i < V; i++) {
          probs_bt[i] = expf(logits_bt[i] - maxval);
          sum += probs_bt[i];
        }
        // note we only loop to V, leaving the padded dimensions
        for (int i = 0; i < V; i++) {
          probs_bt[i] /= sum;
        }
        // for extra super safety we may wish to include this too,
        // forcing the probabilities here to be zero, but it shouldn't matter
        for (int i = V; i < Vp; i++) {
          probs_bt[i] = 0.0f;
        }
      }
    }
  }

  /*
    The cross-entropy C is defined as,

    C = - \sum_b \sum_t \sum_v p(b,t,v)*log(q(b,t,v))

    where,

    p(b,t,v) is the target probability on batch b, time t and vocab token v
    q(b,t,v) is the predicted probability on batch b, time t and vocab token v

    Since, p(b,t,v) is one-hot encoded (i.e. is a delta function), we end up 
    
    C = - \sum_b \sum_t log(q(b,t,v^t(b,t)))

    where,

    v^t(b,t) is the target token index at batch b and time t.

    Often, the predicted probability q(b,t,v) is obtained via softmax over the
    logits (represented by l), where,

    q(b,t,v) = e^{l(b,t,v)-l_{max}(b,v)}/Z(b,t) with,

    l_{max}(b,v) = max(l(b,t,v) | for all v in {0,V})
    Z(b,t) = \sum_v e^{l(b,t,v)-l_{max}(b,v)}

    Now the gradient over the logits is defined as,

    grad(l) = d(C)/d(l(b,t,v))
            = - \sum_b \sum_t \sum_w p(b,t,w)* d(log(q(b,t,w)))/d(l(b,t,v))
            
    now, log(q(b,t,w)) = l(b,t,w) - log(\sum_w e^{l(b,t,w)}), so,
    
    grad(l) = - \sum_b \sum_t \sum_w p(b,t,w) * (\delta_{v,w} - d(log(\sum_w e^{l(b,t,w)}))/d(l(b,t,v)) )

    with d(log(\sum_w e^{l(b,t,w)}))/d(l(b,t,v)) = 1 / (\sum_w e^{l(b,t,w)}) * d(\sum_w e^{l(b,t,w)})/d(l(b,t,v)
)                                                = 1 / (\sum_w e^{l(b,t,w)}) * e^{l(b,t,v)}
                                                 := q(b,t,v)

    grad(l)(b,t,v) = - \sum_b \sum_t \sum_w p(b,t,w) * (\delta_{v,w}-q(b,t,v))
  */
  template<typename index_type, typename value_type>
  class crossentropy
  {
  public:

    static void forward(float* losses,
                        float* probs, int* targets,
                        int B, int T, int Vp);

    static void softmax_backward(float* dlogits,
				 float* dlosses, float* probs, int* targets,
				 int B, int T, int V, int Vp);
    };

  template<typename index_type, typename value_type>
  void crossentropy<index_type, value_type>::forward(float* losses,
                                                     float* probs, int* targets,
                                                     int B, int T, int Vp) {
    //LOG_S(INFO) << "crossentropy::" << __FUNCTION__;

    // output: losses is (B,T) of the individual losses at each position
    // input: probs are (B,T,Vp) of the probabilities
    // input: targets is (B,T) of integers giving the correct index in logits
    for (int b = 0; b < B; b++) {
      for (int t = 0; t < T; t++) {
        // loss = -log(probs[target])
        float* probs_bt = probs + b * T * Vp + t * Vp;
        int ix = targets[b * T + t];
        losses[b * T + t] = -logf(probs_bt[ix]);
      }
    }
  }

  template<typename index_type, typename value_type>
  void crossentropy<index_type, value_type>::softmax_backward(float* dlogits,
							      float* dlosses, float* probs, int* targets,
							      int B, int T, int V, int Vp) {
    // backwards through both softmax and crossentropy
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            float* dlogits_bt = dlogits + b * T * Vp + t * Vp;
            float* probs_bt = probs + b * T * Vp + t * Vp;
            float dloss = dlosses[b * T + t];
            int ix = targets[b * T + t];
            // note we only loop to V, leaving the padded dimensions
            // of dlogits untouched, so gradient there stays at zero
            for (int i = 0; i < V; i++) {
                float p = probs_bt[i];
                float indicator = i == ix ? 1.0f : 0.0f;
                dlogits_bt[i] += (p - indicator) * dloss;
            }
        }
    }
}

  

}

#endif
