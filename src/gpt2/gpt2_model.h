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

  public:
    
    gpt2_model();

    nlohmann::json create_config();

    bool initialise(nlohmann::json config);
    
    bool read(std::string filename);
    bool write(std::string filename);
    
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
    gpt2_activations_type activations;
    gpt2_activations_type activations_grad;
    
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
    int batch_size; // the batch size (B) of current forward pass
    int seq_len; // the sequence length (T) of current forward pass

    int* inputs; // the input tokens for the current forward pass
    int* targets; // the target tokens for the current forward pass

    float mean_loss; // after a forward pass with targets, will be populated with the mean loss
  };

  template<typename index_type, typename value_type>
  gpt2_model<index_type, value_type>::gpt2_model():
    model_config(),
    train_config(),

    weights(),
    weights_grad(),

    activations(),
    activations_grad(),

    m_memory(NULL),
    v_memory(NULL),

    batch_size(1),
    seq_len(1024),
    
    inputs(NULL),
    targets(NULL),

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
    
    weights.initialise(model_config);
    
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
						   std::vector<std::vector<index_type> >& target_tokens)
  {
    
  }
  
}

#endif
