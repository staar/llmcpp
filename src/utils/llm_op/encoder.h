//-*-C++-*-

#ifndef UTILS_LLM_OP_ENCODER_H
#define UTILS_LLM_OP_ENCODER_H

namespace llmcpp
{
  template<typename index_type, typename value_type>
  class encoder
  {
    typedef std::shared_ptr<dense_tensor<index_type, value_type> > dftensor_type;
    typedef std::shared_ptr<shallow_tensor<index_type, value_type> > sftensor_type;

    typedef std::shared_ptr<dense_tensor<index_type, index_type> > ditensor_type;
    
  public:

    static void forward(value_type* out,
                        index_type* inp, value_type* wte, value_type* wpe,
                        index_type B, index_type T, index_type C);

    static void backward(value_type* dwte, value_type* dwpe,
                         value_type* dout, index_type* inp,
                         index_type B, index_type T, index_type C);

    static void forward(dftensor_type out,
                        ditensor_type inp,
			dftensor_type wte,
			dftensor_type wpe);
  }

  template<typename index_type, typename value_type>
  void encoder<index_type, value_type>::forward(value_type* out,
                                                index_type* inp, value_type* wte, value_type* wpe,
                                                index_type B, index_type T, index_type C) {
    //LOG_S(INFO) << "encoder::" << __FUNCTION__;
    
    // out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
    // inp is (B,T) of integers, holding the token ids at each (b,t) position
    // wte is (V,C) of token embeddings, short for "weight token embeddings"
    // wpe is (maxT,C) of position embeddings, short for "weight positional embedding"
    for (index_type b = 0; b < B; b++) {
      for (index_type t = 0; t < T; t++) {
        // seek to the output position in out[b,t,:]
        value_type* out_bt = out + b * T * C + t * C;
        // get the index of the token at inp[b, t]
        index_type ix = inp[b * T + t];
        // seek to the position in wte corresponding to the token
        value_type* wte_ix = wte + ix * C;
        // seek to the position in wpe corresponding to the position
        value_type* wpe_t = wpe + t * C;
        // add the two vectors and store the result in out[b,t,:]
	for (index_type i = 0; i < C; i++) {
          out_bt[i] = wte_ix[i] + wpe_t[i];
        }
      }
    }
  }

  template<typename index_type, typename value_type>
  void encoder<index_type, value_type>::forward(dftensor_type out,
						ditensor_type inp,
						dftensor_type wte,
						dftensor_type wpe)
  {
    for (index_type b = 0; b < inp.size(0); b++) {
      for (index_type t = 0; t < inp.size(1); t++) {

	index_type id = inp(b,t);
	for (index_type i = 0; i < out.size(2); i++) {
	  out(b,t,i) = wte(b,id,i) + wpe(t,i); 
	}
      }
    }
  }
  

  
  template<typename index_type, typename value_type>
  void encoder<index_type, value_type>::backward(value_type* dwte, value_type* dwpe,
                                                 value_type* dout, index_type* inp,
                                                 index_type B, index_type T, index_type C) {
    for (index_type b = 0; b < B; b++) {
      for (index_type t = 0; t < T; t++) {
        value_type* dout_bt = dout + b * T * C + t * C;
        index_type ix = inp[b * T + t];
        value_type* dwte_ix = dwte + ix * C;
        value_type* dwpe_t = dwpe + t * C;
        for (index_type i = 0; i < C; i++) {
          value_type d = dout_bt[i];
          dwte_ix[i] += d;
          dwpe_t[i] += d;
        }
      }
    }
  }

}

#endif
