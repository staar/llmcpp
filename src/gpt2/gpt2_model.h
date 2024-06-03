//-*-C++-*-

#ifndef GPT2_MODEL_H
#define GPT2_MODEL_H

namespace llmcpp
{

  template<typename index_type, typename value_type>
  class gpt2_model
  {
    //typedef llm_tensor<index_type, value_type> llm_tensor_type;
    typedef dense_tensor<index_type, value_type> llm_tensor_type;
    typedef shallow_tensor<index_type, value_type> shl_tensor_type;

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

    bool read_hf_weights(std::string filename);
    
    bool read(std::string filename);
    bool write(std::string filename);

    int get_B() { return model_config.batch_size; }
    int get_maxT() { return model_config.max_seq_len; }

    int to_tensor(std::vector<std::vector<index_type> >& tokens,
                  dense_tensor<index_type, int>& tensor);

    void forward(dense_tensor<index_type, int>& itokens,
                 dense_tensor<index_type, int>& otokens);

    void backward(dense_tensor<index_type, int>& itokens,
                  dense_tensor<index_type, int>& otokens);

  private:

    void set_grad_to_zero();

  public:

    gpt2_model_config model_config;
    gpt2_train_config train_config;

    // the weights (parameters) of the model, and their sizes
    gpt2_weights_type weights;
    gpt2_weights_type weights_grad;

    // the activations of the model, and their sizes
    gpt2_activations_type acts;
    gpt2_activations_type acts_grad;

    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;

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

    if(config["mode"]=="train")
      {
        weights_grad.initialise(model_config);
        acts_grad.initialise(model_config);
      }

    return true;
  }

  /* 
   * Needs to be synced with the `llmcpp/download_weights.py` file
   */
  template<typename index_type, typename value_type>
  bool gpt2_model<index_type, value_type>::read_hf_weights(std::string filename)
  {
    std::ifstream ifs(filename.c_str(), std::ios::binary);

    if(not ifs)
      {
        return false;
      }

    // read the `value_type`
    {
      int32_t dtype_len;
      ifs.read((char*)&dtype_len, sizeof(dtype_len));
      
      LOG_S(INFO) << "dtype-len: "<< dtype_len;
      
      std::string dtype(dtype_len, ' ');
      ifs.read(dtype.data(), sizeof(char)*dtype_len);
      
      LOG_S(INFO) << "dtype: "<< dtype;
    }

    while(!ifs.eof())
      {
	int32_t name_len;
	ifs.read((char*)&name_len, sizeof(name_len));
	
	LOG_S(INFO) << "name-len: "<< name_len;
	
	std::string name(name_len, ' ');
	ifs.read(name.data(), sizeof(char)*name_len);
	
	LOG_S(INFO) << "layer-name: "<< name;
	
	int32_t weights_ndim;
	ifs.read((char*)&weights_ndim, sizeof(weights_ndim));
	
	LOG_S(INFO) << "weights-ndim: "<< weights_ndim;
	
	std::vector<int32_t> weights_dim(weights_ndim,0);
	ifs.read((char*)weights_dim.data(), weights_ndim*sizeof(int32_t));
	
	for(auto _:weights_dim)
	  {
	    LOG_S(INFO) << " => " << _;
	  }
	
	auto tnsr = weights.at(name);
	
	tnsr->initialise(name, weights_dim, weights_dim, false);
	ifs.read((char*)tnsr->ptr(), tnsr->size()*sizeof(value_type));

	if(tnsr->ndim()==2)
	  {
	    LOG_S(INFO) << (*tnsr)(0,1);
	  }
      }
    
    return true;
  }

  template<typename index_type, typename value_type>
  bool gpt2_model<index_type, value_type>::read(std::string filename)
  {
    std::ifstream ifs(filename.c_str(), std::ios::binary);

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
  int gpt2_model<index_type, value_type>::to_tensor(std::vector<std::vector<index_type> >& tokens,
                                                    dense_tensor<index_type, int>& tensor)
  {
    int max_len = 0;

    tensor.to_zero();
    for(int i=0; i<tokens.size(); i++)
      {
        max_len = tokens.at(i).size()>max_len? tokens.at(i).size():max_len;

        for(int j=0; j<tokens.at(i).size(); j++)
          {
            tensor(i,j) = tokens.at(i).at(j);
          }
      }

    return max_len;
  }

  template<typename index_type, typename value_type>
  void gpt2_model<index_type, value_type>::forward(dense_tensor<index_type, int>& itokens,
                                                   dense_tensor<index_type, int>& otokens)
  {
    // convenience parameters (size_t to help prevent int overflow)
    int V = model_config.vocab_size;
    int Vp = model_config.padded_vocab_size;
    int L = model_config.num_layers;
    int NH = model_config.num_heads;
    int C = model_config.channels;

    // training parameters
    int B = itokens.size(0);
    int T = itokens.size(1);

    value_type* residual;

    // encoding goes into residual[0]
    encoder_type::forward(acts.encoded.ptr(), itokens.ptr(), weights.wte.ptr(), weights.wpe.ptr(), B, T, C);

    for(int l=0; l<L; l++)
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
    if (itokens.size(0)==otokens.size(0) and
        itokens.size(1)==otokens.size(1))
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

  template<typename index_type, typename value_type>
  void gpt2_model<index_type, value_type>::backward(dense_tensor<index_type, int>& itokens,
                                                    dense_tensor<index_type, int>& otokens)
  {
    // convenience parameters (size_t to help prevent int overflow)
    int V = model_config.vocab_size;
    int Vp = model_config.padded_vocab_size;
    int L = model_config.num_layers;
    int NH = model_config.num_heads;
    int C = model_config.channels;

    // training parameters
    int B = itokens.size(0);
    int T = itokens.size(1);

    LOG_S(INFO) << "start backward ...";

    // we kick off the chain rule by filling in dlosses with 1.0f/(B*T)
    // technically this is a small, inline backward() pass of calculating
    // total, final loss as the mean over all losses over all (B,T) positions in the batch
    value_type dloss_mean = 1.0f / (B*T);
    for (int b = 0; b < B; b++)
      {
        for (int t = 0; t < T; t++)
          {
            acts_grad.losses(b,t) = dloss_mean;
          }
      }

    crossentropy_type::softmax_backward(acts_grad.logits.ptr(),
                                        acts_grad.losses.ptr(),
                                        acts.probs.ptr(),
                                        otokens.ptr(),
                                        B, T, V, Vp);

    matmul_type::backward(acts_grad.lnf.ptr(),
                          weights_grad.wte.ptr(), NULL,
                          acts_grad.logits.ptr(), acts.lnf.ptr(),
                          weights.wte.ptr(), B, T, C, Vp);

    value_type* residual = acts.residual3.ptr() + (L-1) * B * T * C; // last layer's residual
    value_type* dresidual = acts_grad.residual3.ptr() + (L-1) * B * T * C; // write to last layer's residual

    layernorm_type::backward(dresidual,
                             weights_grad.lnfw.ptr(), weights_grad.lnfb.ptr(), acts_grad.lnf.ptr(),
                             residual, weights.lnfw.ptr(),
                             acts.lnf_mean.ptr(), acts.lnf_rstd.ptr(),
                             B, T, C);

    for (int l = L-1; l >= 0; l--)
      {
        LOG_S(INFO) << "backwarding layer: " << l;

        residual = l == 0 ? acts.encoded.ptr() : acts.residual3.ptr() + (l-1) * B * T * C;
        dresidual = l == 0 ? acts_grad.encoded.ptr() : acts_grad.residual3.ptr() + (l-1) * B * T * C;

        // get the pointers of the weights for this layer
        value_type* l_ln1w = weights.ln1w.ptr() + l * C;
        value_type* l_qkvw = weights.qkvw.ptr() + l * 3*C * C;
        value_type* l_attprojw = weights.attprojw.ptr() + l * C * C;
        value_type* l_ln2w = weights.ln2w.ptr() + l * C;
        value_type* l_fcw = weights.fcw.ptr() + l * 4*C * C;
        value_type* l_fcprojw = weights.fcprojw.ptr() + l * C * 4*C;

        // get the pointers of the gradients of the weights for this layer
        value_type* dl_ln1w = weights_grad.ln1w.ptr() + l * C;
        value_type* dl_ln1b = weights_grad.ln1b.ptr() + l * C;
        value_type* dl_qkvw = weights_grad.qkvw.ptr() + l * 3*C * C;
        value_type* dl_qkvb = weights_grad.qkvb.ptr() + l * 3*C;
        value_type* dl_attprojw = weights_grad.attprojw.ptr() + l * C * C;
        value_type* dl_attprojb = weights_grad.attprojb.ptr() + l * C;
        value_type* dl_ln2w = weights_grad.ln2w.ptr() + l * C;
        value_type* dl_ln2b = weights_grad.ln2b.ptr() + l * C;
        value_type* dl_fcw = weights_grad.fcw.ptr() + l * 4*C * C;
        value_type* dl_fcb = weights_grad.fcb.ptr() + l * 4*C;
        value_type* dl_fcprojw = weights_grad.fcprojw.ptr() + l * C * 4*C;
        value_type* dl_fcprojb = weights_grad.fcprojb.ptr() + l * C;

        // get the pointers of the activations for this layer
        value_type* l_ln1 = acts.ln1.ptr() + l * B * T * C;
        value_type* l_ln1_mean = acts.ln1_mean.ptr() + l * B * T;
        value_type* l_ln1_rstd = acts.ln1_rstd.ptr() + l * B * T;
        value_type* l_qkv = acts.qkv.ptr() + l * B * T * 3*C;
        value_type* l_atty = acts.atty.ptr() + l * B * T * C;
        value_type* l_att = acts.att.ptr() + l * B * NH * T * T;
        value_type* l_residual2 = acts.residual2.ptr() + l * B * T * C;
        value_type* l_ln2 = acts.ln2.ptr() + l * B * T * C;
        value_type* l_ln2_mean = acts.ln2_mean.ptr() + l * B * T;
        value_type* l_ln2_rstd = acts.ln2_rstd.ptr() + l * B * T;
        value_type* l_fch = acts.fch.ptr() + l * B * T * 4*C;
        value_type* l_fch_gelu = acts.fch_gelu.ptr() + l * B * T * 4*C;

        // get the pointers of the gradients of the activations for this layer
        value_type* dl_ln1 = acts_grad.ln1.ptr() + l * B * T * C;
        value_type* dl_qkv = acts_grad.qkv.ptr() + l * B * T * 3*C;
        value_type* dl_atty = acts_grad.atty.ptr() + l * B * T * C;
        value_type* dl_preatt = acts_grad.preatt.ptr() + l * B * NH * T * T;
        value_type* dl_att = acts_grad.att.ptr() + l * B * NH * T * T;
        value_type* dl_attproj = acts_grad.attproj.ptr() + l * B * T * C;
        value_type* dl_residual2 = acts_grad.residual2.ptr() + l * B * T * C;
        value_type* dl_ln2 = acts_grad.ln2.ptr() + l * B * T * C;
        value_type* dl_fch = acts_grad.fch.ptr() + l * B * T * 4*C;
        value_type* dl_fch_gelu = acts_grad.fch_gelu.ptr() + l * B * T * 4*C;
        value_type* dl_fcproj = acts_grad.fcproj.ptr() + l * B * T * C;
        value_type* dl_residual3 = acts_grad.residual3.ptr() + l * B * T * C;

        // backprop this layer
        residual_type::backward(dl_residual2, dl_fcproj, dl_residual3, B*T*C);
        matmul_type::backward(dl_fch_gelu, dl_fcprojw, dl_fcprojb, dl_fcproj, l_fch_gelu, l_fcprojw, B, T, 4*C, C);
        gelu_type::backward(dl_fch, l_fch, dl_fch_gelu, B*T*4*C);
        matmul_type::backward(dl_ln2, dl_fcw, dl_fcb, dl_fch, l_ln2, l_fcw, B, T, C, 4*C);
        layernorm_type::backward(dl_residual2, dl_ln2w, dl_ln2b, dl_ln2, l_residual2, l_ln2w, l_ln2_mean, l_ln2_rstd, B, T, C);
        residual_type::backward(dresidual, dl_attproj, dl_residual2, B*T*C);
        matmul_type::backward(dl_atty, dl_attprojw, dl_attprojb, dl_attproj, l_atty, l_attprojw, B, T, C, C);
        attention_type::backward(dl_qkv, dl_preatt, dl_att, dl_atty, l_qkv, l_att, B, T, C, NH);
        matmul_type::backward(dl_ln1, dl_qkvw, dl_qkvb, dl_qkv, l_ln1, l_qkvw, B, T, C, 3*C);
        layernorm_type::backward(dresidual, dl_ln1w, dl_ln1b, dl_ln1, residual, l_ln1w, l_ln1_mean, l_ln1_rstd, B, T, C);
      }
    LOG_S(INFO) << "done backwarding layers ...";

    encoder_type::backward(weights_grad.wte.ptr(), weights_grad.wpe.ptr(), acts_grad.encoded.ptr(), itokens.ptr(), B, T, C);
  }

  template<typename index_type, typename value_type>
  void gpt2_model<index_type, value_type>::set_grad_to_zero()
  {
    weights_grad.to_zero();
    acts_grad.to_zero();
  }

}

#endif
