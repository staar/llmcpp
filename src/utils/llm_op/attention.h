//-*-C++-*-

#ifndef UTILS_LLM_OP_ATTENTION_H
#define UTILS_LLM_OP_ATTENTION_H

namespace llmcpp
{
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

    static void test_forward();

    static void test_backward();

  private:

    static void forward_orig(float* out, float* preatt, float* att,
			     float* inp,
			     int B, int T, int C, int NH);
    
    static void forward_blas(float* out, float* preatt, float* att,
			     float* inp,
			     int B, int T, int C, int NH);

    static void backward_orig(float* dinp, float* dpreatt, float* datt,
			      float* dout, float* inp, float* att,
			      int B, int T, int C, int NH);

    static void backward_blas(float* dinp, float* dpreatt, float* datt,
			      float* dout, float* inp, float* att,
			      int B, int T, int C, int NH);
  };

  template<typename index_type, typename value_type>
  void attention<index_type, value_type>::forward(float* out, float* preatt, float* att,
                                                  float* inp,
                                                  int B, int T, int C, int NH) {
    forward_blas(out, preatt, att, inp, B, T, C, NH);
  }
  
  template<typename index_type, typename value_type>
  void attention<index_type, value_type>::forward_orig(float* out, float* preatt, float* att,
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
              preatt_bth[t2] = 0.0f; // I added
	      att_bth[t2] = 0.0f;
            }
          }

          // pass 4: accumulate weighted values into the output of attention
          float* out_bth = out + b * T * C + t * C + h * hs;
          for (int i = 0; i < hs; i++)
	    {
	      out_bth[i] = 0.0f;
	    }

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
  void attention<index_type, value_type>::forward_blas(float* out, float* preatt, float* att,
						       float* inp,
						       int B, int T, int C, int NH) {
    //LOG_S(INFO) << "attantion::" << __FUNCTION__;

    // input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    // preatt, (B, NH, T, T). NH = number of heads, T = sequence length
    // att are (B, NH, T, T). NH = number of heads, T = sequence length
    // that holds the pre-attention and post-attention scores (used in backward)
    // output is (B, T, C)
    // attention is the only layer that mixes information across time
    // every other operation is applied at every (b,t) position independently
    // (and of course, no layer mixes information across batch)

    int C3 = C*3;
    int HS = C / NH; // head size

    if(NH*HS!=C)
      {
	LOG_S(FATAL) << "NH:" << NH << ", HS:" << HS << ", C:" << C;
      }

    float scale = 1.0 / sqrtf(HS);

    // compute the preatt
    for (int b = 0; b < B; b++) {
      for (int h = 0; h < NH; h++) {
	//for (int t = 0; t < T; t++) {

	int t1=0, t2=0;
	
	float* query_bh = inp + b * T * C3 + t1 * C3 + h * HS + 0*C; // +0*C because it's query
	float* key_bh   = inp + b * T * C3 + t2 * C3 + h * HS + 1*C; // +1*C because it's key

	float* preatt_bh = preatt + b*NH*T*T + h*T*T;
	float* att_bh    =    att + b*NH*T*T + h*T*T;  

	value_type alpha = scale, beta = 0.0;
	blas::gemm<value_type>::execute('T', 'N', T, T, HS, alpha,
					key_bh, 3*C,
					query_bh, 3*C, 
					beta, preatt_bh, T);

	// because this is a causal LLM
	for (int t1 = 0; t1 < T; t1++) {
	  for (int t2 = t1+1; t2 < T; t2++) {
	    preatt_bh[t1*T+t2] = 0;
	  }
	}

	// compute the softmax
	for (int t1 = 0; t1 < T; t1++) {

	  value_type max = preatt_bh[t1*T+0];
	  for (int t2 = 0; t2 <= t1; t2++) {
	    max = std::max(preatt_bh[t1*T+t2], max);
	  }

	  value_type sum = 1.e-6;
	  for (int t2 = 0; t2 <= t1; t2++) {
	    att_bh[t1*T+t2] = std::exp(preatt_bh[t1*T+t2]-max);
	    sum += att_bh[t1*T+t2];
	  }

	  for (int t2 = 0; t2 <= t1; t2++) {
	    att_bh[t1*T+t2] /= sum;
	  }
	}

	// accumulate into value
	// col-major: out[HS, T] = val_bh[HS, T] * att[T, T]
	
	float* val_bh = inp + b*T*3*C + 2*C + h*HS;  
	float* out_bh = out + b*T*C         + h*HS;

	alpha = 1.0;
	blas::gemm<value_type>::execute('N', 'N', HS, T, T, alpha,
					val_bh, 3*C, 
					att_bh, T,
					beta, out_bh, C);	
      }
    }
  }

  template<typename index_type, typename value_type>
  void attention<index_type, value_type>::attention<index_type, value_type>::test_forward()
  {
    LOG_S(INFO) << __FUNCTION__ << " for attention";
    
    int B=13, T=233, C=56, NH=8;

    llm_tensor<int, float> inp, preatt1, preatt2, att1, att2, out1, out2;
    
    inp.initialise("inp", {B, T, 3*C}, false).to_rand();

    preatt1.initialise("preatt1", {B, NH, T, T}, false).to_rand();
    preatt2.initialise("preatt2", {B, NH, T, T}, false).to_rand();
    
    att1.initialise("att1", {B, NH, T, T}, false).to_zero();
    att2.initialise("att2", {B, NH, T, T}, false).to_zero();
    
    out1.initialise("out1", {B, T, C}, false).to_zero();
    out2.initialise("out2", {B, T, C}, false).to_zero();

    value_type max_diff = 0;

    /*
    for(int i=0; i<10; i++)
      {
	LOG_S(INFO) <<  preatt1(0, 0, i, 0) << "\t" << preatt2(0, 0, i, 0);
      }
    max_diff = preatt1.max_diff(preatt2);
    LOG_S(INFO) << "max-diff: " << max_diff;
    */
    
    forward_orig(out1.ptr(), preatt1.ptr(), att1.ptr(), inp.ptr(), B, T, C, NH);
    forward_blas(out2.ptr(), preatt2.ptr(), att2.ptr(), inp.ptr(), B, T, C, NH);

    /*
    for(int i=0; i<5; i++)
      {
	for(int j=0; j<5; j++)
	  {
	    LOG_S(INFO) << i << "," << j << ": " << preatt1(0, 1, i, j) << "\t" << preatt2(0, 1, i, j);
	  }
      }
    */
    
    max_diff = preatt1.max_diff(preatt2);
    LOG_S(INFO) << "max-diff pre-attn: " << max_diff;

    max_diff = att1.max_diff(att2);
    LOG_S(INFO) << "max-diff attn: " << max_diff;

    max_diff = out1.max_diff(out2);
    LOG_S(INFO) << "max-diff out: " << max_diff;
  }

  template<typename index_type, typename value_type>
  void attention<index_type, value_type>::backward(float* dinp, float* dpreatt, float* datt,
                                                   float* dout, float* inp, float* att,
                                                   int B, int T, int C, int NH) {
    backward_blas(dinp, dpreatt, datt, dout, inp, att, B, T, C, NH);
  }
  
  template<typename index_type, typename value_type>
  void attention<index_type, value_type>::backward_orig(float* dinp, float* dpreatt, float* datt,
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
  void attention<index_type, value_type>::backward_blas(float* dinp, float* dpreatt, float* datt,
							float* dout, float* inp, float* att,
							int B, int T, int C, int NH) {
    // inp/dinp are (B, T, 3C) Q,K,V
    // att/datt/dpreatt are (B, NH, T, T)
    // dout is (B, T, C)
    int C3 = C*3;
    int HS = C / NH; // head size
    
    float scale = 1.0 / sqrtf(HS);

    for (int b = 0; b < B; b++) {
      for (int h = 0; h < NH; h++) {
	
	// backward pass 4, through the value accumulation

	//float* datt_bth = datt + b*NH*T*T + h*T*T + t1*T;
	float* datt_bh = datt + b*NH*T*T + h*T*T;
	//float* dout_bth = dout + b * T * C + t1 * C + h * hs;
	float* dout_bh = dout + b * T * C + h * HS;
	//float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C*2; // +C*2 because it's value
	float* value_bh = inp + b * T * C3 + h * HS + C*2; // +C*2 because it's value
	
	value_type alpha = 1.0, beta = 0.0;
	blas::gemm<value_type>::execute('T', 'N', T, T, HS, alpha,
					value_bh, 3*C,
					dout_bh, C, 
					beta, datt_bh, T);
	
	// for causality ...
	for (int t1 = 0; t1 < T; t1++) {
	  for (int t2 = t1+1; t2 < T; t2++) {
	    datt_bh[t1*T+t2] = 0;
	  }
	}

	//float* att_bth = att + b*NH*T*T + h*T*T + t1*T;
	float* att_bh = att + b*NH*T*T + h*T*T;
	
	//float* dvalue_t2 = dinp + b * T * C3 + t2 * C3 + h * hs + C*2;
	float* dvalue_bh = dinp + b * T * C3 + h * HS + C*2;

	//value_type alpha = 1.0, beta = 0.0;
	blas::gemm<value_type>::execute('N', 'T', HS, T, T, alpha,
					dout_bh, C, 
					att_bh, T,
					beta, dvalue_bh, 3*C);

	// backward pass 2 & 3, the softmax
	// note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
	//float* dpreatt_bth = dpreatt + b*NH*T*T + h*T*T + t1*T;
	float* dpreatt_bh = dpreatt + b*NH*T*T + h*T*T;

	std::vector<value_type> tmp(T, 0);
	for (int t1 = 0; t1 < T; t1++) {
	  tmp.at(t1) = 0.0;
	  for (int t2 = 0; t2 <=t1; t2++) {
	    tmp.at(t1) += att_bh[t1*T+t2]*datt_bh[t1*T+t2];
	  }
	}

	for (int t1 = 0; t1 < T; t1++) {
	  for (int t3 = 0; t3 <= t1; t3++) {
	    auto delta = datt_bh[t1*T+t3]-tmp.at(t1);
	    dpreatt_bh[t1*T+t3] = att_bh[t1*T+t3]*delta;
	  }
	}
	
	// for causality ...
	for (int t1 = 0; t1 < T; t1++) {
	  for (int t2 = t1+1; t2 < T; t2++) {
	    dpreatt_bh[t1*T+t2] = 0;
	  }
	}

	// backward pass 1, the query @ key matmul
	
	//float* dquery_t = dinp + b * T * C3 + t1 * C3 + h * HS;
	float* dquery_bh = dinp + b * T * C3 + h * HS;
	//float* query_t = inp + b * T * C3 + t1 * C3 + h * HS;
	float* query_bh = inp + b * T * C3 + h * HS;
	
	//float*  key_t2 = inp + b * T * C3 + t2 * C3 + h * HS + C; // +C because it's key
	float*  key_bh = inp + b * T * C3 + h * HS + C; // +C because it's key
	//float* dkey_t2 = dinp + b * T * C3 + t2 * C3 + h * HS + C; // +C because it's key
	float* dkey_bh = dinp + b * T * C3 + h * HS + C; // +C because it's key
	
	blas::gemm<value_type>::execute('N', 'N', HS, T, T, scale,
					key_bh, 3*C,
					dpreatt_bh, T, 
					beta, dquery_bh, 3*C);
		
	blas::gemm<value_type>::execute('N', 'T', HS, T, T, scale,
					query_bh, 3*C,
					dpreatt_bh, T, 
					beta, dkey_bh, 3*C);
      }
    }
  }

  template<typename index_type, typename value_type>
  void attention<index_type, value_type>::attention<index_type, value_type>::test_backward()
  {
    LOG_S(INFO) << __FUNCTION__ << " for attention";
    
    int B=13, T=233, C=56, NH=8;

    //float* dinp, float* dpreatt, float* datt,
    //float* dout, float* inp, float* att,
    
    llm_tensor<int, float> dout, inp, att, dinp1, dinp2, dpreatt1, dpreatt2, datt1, datt2;

    dout.initialise("dout", {B, T, 3*C}, false).to_rand();
    inp.initialise("inp", {B, T, 3*C}, false).to_rand();
    att.initialise("att", {B, NH, T, T}, false).to_rand();

    // ensure attention is causal ...
    for (int b = 0; b < B; b++) {
      for (int h = 0; h < NH; h++) {
	for (int t1 = 0; t1 < T; t1++) {
	  for (int t2 = t1+1; t2 < T; t2++) {
	    att(b, h, t1, t2) = 0;
	  }
	}
      }
    }
    
    dpreatt1.initialise("preatt1", {B, NH, T, T}, false).to_zero();
    dpreatt2.initialise("preatt2", {B, NH, T, T}, false).to_zero();
    
    datt1.initialise("datt1", {B, NH, T, T}, false).to_zero();
    datt2.initialise("datt2", {B, NH, T, T}, false).to_zero();
    
    dinp1.initialise("dinp1", {B, T, 3*C}, false).to_zero();
    dinp2.initialise("dinp2", {B, T, 3*C}, false).to_zero();

    value_type max_diff = 0;

    /*
      for(int i=0; i<10; i++)
      {
      LOG_S(INFO) <<  preatt1(0, 0, i, 0) << "\t" << preatt2(0, 0, i, 0);
      }
      max_diff = preatt1.max_diff(preatt2);
      LOG_S(INFO) << "max-diff: " << max_diff;
    */
    
    backward_orig(dinp1.ptr(), dpreatt1.ptr(), datt1.ptr(), dout.ptr(), inp.ptr(), att.ptr(), B, T, C, NH);
    backward_blas(dinp2.ptr(), dpreatt2.ptr(), datt2.ptr(), dout.ptr(), inp.ptr(), att.ptr(), B, T, C, NH);

    /*
    for(int i=0; i<5; i++)
      {
	for(int j=0; j<5; j++)
	  {
	    LOG_S(INFO) << i << "," << j << ": "
			<< datt1(1, 1, i, j) << "\t"
			<< datt2(1, 1, i, j) << "\t"
			<< std::abs(datt2(1, 1, i, j)-datt1(1, 1, i, j));
	  }
      }
    */
    
    max_diff = datt1.max_reldiff(datt2);
    LOG_S(INFO) << "max-diff dattn: " << max_diff;
    
    max_diff = dpreatt1.max_diff(dpreatt2);
    LOG_S(INFO) << "max-diff dpre-attn: " << max_diff;

    /*
    for (int b = 0; b < B; b++) 
      {
	for(int i=0; i<T; i++)
	  {
	    for(int j=0; j<3*C; j++)
	      {
		if(std::abs(dinp2(b, i, j)-dinp1(b, i, j))>1.e-3)
		  {
		    LOG_S(WARNING) << i << "," << j << ": " << std::setw(8)
				   << dinp1(b, i, j) << "\t" << std::setw(8)
				   << dinp2(b, i, j) << "\t" << std::setw(8)
				   << std::abs(dinp2(b, i, j)-dinp1(b, i, j));
		  }
		else
		  {
		    //LOG_S(INFO) << i << "," << j << ": "
		    //<< dinp1(1, i, 2*C+j) << "\t"
		    //<< dinp2(1, i, 2*C+j) << "\t"
		    //<< std::abs(dinp2(1, i, 2*C+j)-dinp1(1, i, 2*C+j));
		  }
	      }
	  }
      }
    */
    
    max_diff = dinp1.max_diff(dinp2);
    LOG_S(INFO) << "max-diff dinp: " << max_diff;
  }
}

#endif
