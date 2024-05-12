//-*-C++-*-

#ifndef GPT2_MODEL_H
#define GPT2_MODEL_H

namespace llmcpp
{

  template<typename index_type, typename value_type>
  class gpt2_model
  {
    typedef llm_tensor<index_type, value_type> llm_tensor_type;

    typedef gpt2_weights<index_type, value_type> gpt2_weights_type;
    typedef gpt2_activations<index_type, value_type> gpt2_activations_type;

    typedef encoder<index_type, value_type> encoder_type;
    typedef matmul<index_type, value_type> matmul_type;
    typedef layernorm<index_type, value_type> layernorm_type;
    typedef softmax<index_type, value_type> softmax_type;
    typedef crossentropy<index_type, value_type> crossentropy_type;
    typedef attention<index_type, value_type> attention_type;
    typedef residual<index_type, value_type> residual_type;
    typedef gelu<index_type, value_type> gelu_type;

  public:

    gpt2_model();

    nlohmann::json create_config();

    bool initialise(nlohmann::json config);

    bool read(std::string filename);
    bool write(std::string filename);

    int get_B() { return model_config.batch_size; }
    int get_maxT() { return model_config.max_seq_len; }

    void forward(std::vector<std::vector<index_type> >& input_tokens,
                 std::vector<std::vector<index_type> >& target_tokens);

  public:

    gpt2_model_config model_config;
    gpt2_train_config train_config;

    // the weights (parameters) of the model, and their sizes
    gpt2_weights_type weights;
    gpt2_weights_type weights_grad;

    //ParameterTensors params;
    //size_t param_sizes[NUM_PARAMETER_TENSORS];
    //float* params_memory;
    //size_t num_parameters;

    // gradients of the weights
    //ParameterTensors grads;
    //float* grads_memory;

    // the activations of the model, and their sizes
    gpt2_activations_type acts;
    gpt2_activations_type acts_grad;

    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;

    //ActivationTensors acts;
    //size_t act_sizes[NUM_ACTIVATION_TENSORS];
    //float* acts_memory;
    //size_t num_activations;

    // gradients of the activations
    //ActivationTensors grads_acts;
    //float* grads_acts_memory;

    // other run state configuration
    //int batch_size; // the batch size (B) of current forward pass
    //int seq_len; // the sequence length (T) of current forward pass

    //int* inputs; // the input tokens for the current forward pass
    //int* targets; // the target tokens for the current forward pass

    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
  };

  template<typename index_type, typename value_type>
  gpt2_model<index_type, value_type>::gpt2_model():
    model_config(),
    train_config(),

    weights(),
    weights_grad(),

    acts(),
    acts_grad(),

    m_memory(NULL),
    v_memory(NULL),

    //batch_size(1),
    //seq_len(1024),

    //inputs(NULL),
    //targets(NULL),

    mean_loss(-1.0)
  {}

  template<typename index_type, typename value_type>
  nlohmann::json gpt2_model<index_type, value_type>::create_config()
  {
    nlohmann::json config = nlohmann::json::object({});

    config["model"] = model_config.to_json();
    config["train"] = train_config.to_json();

    return config;
  }

  template<typename index_type, typename value_type>
  bool gpt2_model<index_type, value_type>::initialise(nlohmann::json config)
  {
    LOG_S(INFO) << __FUNCTION__;

    model_config.from_json(config["model"]);
    
    weights.initialise(model_config);
    acts.initialise(model_config);

    return true;
  }

  template<typename index_type, typename value_type>
  bool gpt2_model<index_type, value_type>::read(std::string filename)
  {
    std::ofstream ifs(filename.c_str(), std::ios::binary);

    if(not ifs)
      {
        return false;
      }

    ifs >> weights;
    return true;
  }

  template<typename index_type, typename value_type>
  bool gpt2_model<index_type, value_type>::write(std::string filename)
  {
    std::ofstream ofs(filename.c_str(), std::ios::binary);

    if(not ofs)
      {
        return false;
      }

    ofs << weights;
    return true;
  }

  template<typename index_type, typename value_type>
  void gpt2_model<index_type, value_type>::forward(std::vector<std::vector<index_type> >& input_tokens,
                                                   std::vector<std::vector<index_type> >& output_tokens)
  {
    // convenience parameters (size_t to help prevent int overflow)
    int V = model_config.vocab_size;
    int Vp = model_config.padded_vocab_size;
    int L = model_config.num_layers;
    int NH = model_config.num_heads;
    int C = model_config.channels;

    // training parameters
    int B = input_tokens.size();
    int T = input_tokens.at(0).size();

    llm_tensor<index_type, int> itokens("itokens", {B, T}, false);
    llm_tensor<index_type, int> otokens("otokens", {B, T}, false);

    for(int i=0; i<B; i++)
      {
        for(int j=0; j<T; j++)
          {
            itokens(i,j) = input_tokens.at(i).at(j);
          }
      }

    if(output_tokens.size()==B)
      {
	for(int i=0; i<B; i++)
	  {
	    for(int j=0; j<T; j++)
	      {
		otokens(i,j) = output_tokens.at(i).at(j);
	      }
	  }
      }
    
    value_type* residual;

    // encoding goes into residual[0]
    encoder_type::forward(acts.encoded.ptr(), itokens.ptr(), weights.wte.ptr(), weights.wpe.ptr(), B, T, C);

    for (int l=0; l<L; l++)
      {
	LOG_S(INFO) << "forwarding layer: " << l;

      residual = l == 0 ? acts.encoded.ptr() : acts.residual3.ptr() + (l-1) * B * T * C;

      // get the pointers of the weights for this layer
      value_type* l_ln1w = weights.ln1w.ptr() + l * C;
      value_type* l_ln1b = weights.ln1b.ptr() + l * C;
      value_type* l_qkvw = weights.qkvw.ptr() + l * 3*C * C;
      value_type* l_qkvb = weights.qkvb.ptr() + l * 3*C;
      value_type* l_attprojw = weights.attprojw.ptr() + l * C * C;
      value_type* l_attprojb = weights.attprojb.ptr() + l * C;
      value_type* l_ln2w = weights.ln2w.ptr() + l * C;
      value_type* l_ln2b = weights.ln2b.ptr() + l * C;
      value_type* l_fcw = weights.fcw.ptr() + l * 4*C * C;
      value_type* l_fcb = weights.fcb.ptr() + l * 4*C;
      value_type* l_fcprojw = weights.fcprojw.ptr() + l * C * 4*C;
      value_type* l_fcprojb = weights.fcprojb.ptr() + l * C;

      // get the pointers of the activations for this layer
      value_type* l_ln1 = acts.ln1.ptr() + l * B * T * C;
      value_type* l_ln1_mean = acts.ln1_mean.ptr() + l * B * T;
      value_type* l_ln1_rstd = acts.ln1_rstd.ptr() + l * B * T;
      value_type* l_qkv = acts.qkv.ptr() + l * B * T * 3*C;
      value_type* l_atty = acts.atty.ptr() + l * B * T * C;
      value_type* l_preatt = acts.preatt.ptr() + l * B * NH * T * T;
      value_type* l_att = acts.att.ptr() + l * B * NH * T * T;
      value_type* l_attproj = acts.attproj.ptr() + l * B * T * C;
      value_type* l_residual2 = acts.residual2.ptr() + l * B * T * C;
      value_type* l_ln2 = acts.ln2.ptr() + l * B * T * C;
      value_type* l_ln2_mean = acts.ln2_mean.ptr() + l * B * T;
      value_type* l_ln2_rstd = acts.ln2_rstd.ptr() + l * B * T;
      value_type* l_fch = acts.fch.ptr() + l * B * T * 4*C;
      value_type* l_fch_gelu = acts.fch_gelu.ptr() + l * B * T * 4*C;
      value_type* l_fcproj = acts.fcproj.ptr() + l * B * T * C;
      value_type* l_residual3 = acts.residual3.ptr() + l * B * T * C;

      // now do the forward pass
      layernorm_type::forward(l_ln1, l_ln1_mean, l_ln1_rstd, residual, l_ln1w, l_ln1b, B, T, C);
      matmul_type::forward(l_qkv, l_ln1, l_qkvw, l_qkvb, B, T, C, 3*C);

      attention_type::forward(l_atty, l_preatt, l_att, l_qkv, B, T, C, NH);

      matmul_type::forward(l_attproj, l_atty, l_attprojw, l_attprojb, B, T, C, C);
      residual_type::forward(l_residual2, residual, l_attproj, B*T*C);

      layernorm_type::forward(l_ln2, l_ln2_mean, l_ln2_rstd, l_residual2, l_ln2w, l_ln2b, B, T, C);

      matmul_type::forward(l_fch, l_ln2, l_fcw, l_fcb, B, T, C, 4*C);
      gelu_type::forward(l_fch_gelu, l_fch, B*T*4*C);

      matmul_type::forward(l_fcproj, l_fch_gelu, l_fcprojw, l_fcprojb, B, T, 4*C, C);
      residual_type::forward(l_residual3, l_residual2, l_fcproj, B*T*C);
    }
    LOG_S(INFO) << "done looping layers ...";
      
    residual = acts.residual3.ptr() + (L-1) * B * T * C; // last residual is in residual3
    layernorm_type::forward(acts.lnf.ptr(), acts.lnf_mean.ptr(), acts.lnf_rstd.ptr(),
			    residual, weights.lnfw.ptr(), weights.lnfb.ptr(),
			    B, T, C);

    matmul_type::forward(acts.logits.ptr(), acts.lnf.ptr(), weights.wte.ptr(), NULL, B, T, C, Vp);
    softmax_type::forward(acts.probs.ptr(), acts.logits.ptr(), B, T, V, Vp);

    // also forward the cross-entropy loss function if we have the targets
    if (output_tokens.size()==B)
      {
	LOG_S(INFO) << "compute loss ...";
	crossentropy_type::forward(acts.losses.ptr(), acts.probs.ptr(), otokens.ptr(), B, T, Vp);
	
	// for convenience also evaluate the mean loss
	float mean_loss = 0.0f;
	for (int b=0; b<B; b++)
	  {
	    for (int t=0; t<T; t++)
	      {
		mean_loss += acts.losses(b,t);
	      }
	  }
	
	mean_loss /= (B*T);
	
	this->mean_loss = mean_loss;
	LOG_S(INFO) << "loss: " << this->mean_loss;
      }
    else
      {
	// if we don't have targets, we don't have a loss
	this->mean_loss = -1.0f;
      }
    
  }
  
}

#endif
