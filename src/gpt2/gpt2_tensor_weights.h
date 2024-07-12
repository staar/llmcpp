//-*-C++-*-

#ifndef GPT2_TENSOR_WEIGHTS_H
#define GPT2_TENSOR_WEIGHTS_H

#include <utils/llm_tensor.h>

namespace llmcpp
{
  template<typename index_type, typename value_type>
  class gpt2_weights
  {
  public:
    
    typedef dense_tensor<index_type, value_type> llm_tensor_type;

    const static inline std::vector<std::string> names =
      {
       "token-embedding",
       "pos-embedding",
       "ln_1.weight", "ln_1.bias",
       "attn.qkv.weight", "attn.qkv.bias",
       "attn.proj.weight", "attn.proj.bias",
       "ln_2.weight", "ln_2.bias",
       "mlp.fc.weight", "mlp.fc.bias",
       "mlp.proj.weight", "mlp.proj.bias",
       "ln_f.weight", "ln_f.bias"
      };

  public:

    gpt2_weights() {}

    gpt2_weights(index_type V,
                 index_type C,
                 index_type L,
                 index_type maxT);

    gpt2_weights(const nlohmann::json& config);

    index_type get_V() { return V; }
    index_type get_Vp() { return Vp; }
    index_type get_C() { return C; }
    index_type get_L() { return L; }
    index_type get_maxT() { return maxT; }
    
    bool initialise(gpt2_model_config& config);

    bool verify_dims(gpt2_model_config& config) { return true; }
    
    std::shared_ptr<llm_tensor_type> at(std::string name, bool strict=true);

    value_type* ptr(std::string name);
    value_type* ptr(std::string name, index_type layer);

    //friend std::ofstream& operator<<(std::ofstream& ofs, const gpt2_weights<index_type, value_type>& weights);
    //friend std::ifstream& operator>>(std::ifstream& ifs, gpt2_weights<index_type, value_type>& weights);

  public:

    index_type V, Vp, C, L, maxT;

    /*
    //std::shared_ptr<llm_tensor_type> wte; // (V, C)
    llm_tensor_type wte; // (V, C)
    llm_tensor_type wpe; // (maxT, C)
    llm_tensor_type ln1w; // (L, C)
    llm_tensor_type ln1b; // (L, C)
    llm_tensor_type qkvw; // (L, 3*C, C)
    llm_tensor_type qkvb; // (L, 3*C)
    llm_tensor_type attprojw; // (L, C, C)
    llm_tensor_type attprojb; // (L, C)
    llm_tensor_type ln2w; // (L, C)
    llm_tensor_type ln2b; // (L, C)
    llm_tensor_type fcw; // (L, 4*C, C)
    llm_tensor_type fcb; // (L, 4*C)
    llm_tensor_type fcprojw; // (L, C, 4*C)
    llm_tensor_type fcprojb; // (L, C)
    llm_tensor_type lnfw; // (C)
    llm_tensor_type lnfb; // (C)
    */
    
    std::map<std::string, std::shared_ptr<llm_tensor_type> > tensors;
  };

  template<typename index_type, typename value_type>
  gpt2_weights<index_type, value_type>::gpt2_weights(index_type V,
                                                     index_type C,
                                                     index_type L,
                                                     index_type maxT):
    V(V), Vp(V), C(C), L(L), maxT(maxT),

    /*
    //wte(std::make_shared<llm_tensor_type>("token-embedding", {V, C})),
    wte("token-embedding", {V, C}),
    wpe("pos-embedding", {maxT, C}),
    ln1w("ln_1.weight", {L, C}),
    ln1b("ln_1.bias" , {L, C}),
    qkvw("attn.qkv.weights" , {L, 3*C, C}),
    qkvb("attn.qkv.bias" , {L, 3*C}),
    attprojw("attn.proj.weights" , {L, C, C}),
    attprojb("attn.proj.bias" , {L, C}),
    ln2w("ln_2.weight" , {L, C}),
    ln2b("ln_2.bias" , {L, C}),
    fcw("mlp.fc.weight" , {L, 4*C, C}),
    fcb("mlp.fc.bias" , {L, 4*C}),
    fcprojw("mlp.projw.eight" , {L, C, 4*C}),
    fcprojb("mlp.proj.bias" , {L, C}),
    lnfw("ln_f.weight" , {C}),
    lnfb("ln_f.bias" , {C}),
    */
    
    tensors({})
  {}

  template<typename index_type, typename value_type>
  std::shared_ptr<typename gpt2_weights<index_type, value_type>::llm_tensor_type> gpt2_weights<index_type, value_type>::at(std::string name, bool strict)
  {
    //assert(find(names.begin(), names.end(), name)!=names.end());
    
    if((not strict) and tensors.count(name)==0) // insert new tensor
      {
        tensors[name] = std::make_shared<llm_tensor_type>(name);
      }
    else if(strict and tensors.count(name)==0)
      {
	LOG_S(ERROR) << "no tensor with name: " << name;
      }
    
    return tensors.at(name);
  }

  template<typename index_type, typename value_type>
  bool gpt2_weights<index_type, value_type>::initialise(gpt2_model_config& config)
  {
    LOG_S(INFO) << "initialising the GPT2 weights ...";

    V = config.vocab_size;
    Vp = config.padded_vocab_size;
    C = config.channels;
    L = config.num_layers;
    maxT = config.max_seq_len;

    for(auto name:names)
      {
	tensors[name] = std::make_shared<llm_tensor_type>(name);
      }

    tensors.at("token-embedding")->initialise("token-embedding", {V, C}, {Vp, C}, false);
    tensors.at("pos-embedding")->initialise("pos-embedding", {maxT, C}, false);

    tensors.at("ln1w")->initialise("ln1w", {L, C}, false);
    tensors.at("ln1b")->initialise("ln1b" , {L, C}, false);

    tensors.at("qkvw")->initialise("qkv-weights" , {L, 3*C, C}, false);
    tensors.at("qkvb")->initialise("qkv-bias" , {L, 3*C}, false);

    tensors.at("attprojw")->initialise("attention-weights" , {L, C, C}, false);
    tensors.at("attprojb")->initialise("attention-bias" , {L, C}, false);

    tensors.at("ln2w")->initialise("ln2w" , {L, C}, false);
    tensors.at("ln2b")->initialise("ln2b" , {L, C}, false);

    tensors.at("fcw")->initialise("fcw" , {L, 4*C, C}, false);
    tensors.at("fcb")->initialise("fcb" , {L, 4*C}, false);

    tensors.at("fcprojw")->initialise("fcprojw" , {L, C, 4*C}, false);
    tensors.at("fcprojb")->initialise("fcprojb" , {L, C}, false);

    tensors.at("lnfw")->initialise("lnfw" , {C}, false);
    tensors.at("lnfb")->initialise("lnfb" , {C}, false);
    
    //LOG_S(INFO) << "V = " << V << << "Vp = " << Vp << "; C = " << C << "; L = " << L << "; maxT = " << maxT;
    /*
    wte.initialise("weight-embedding", {V, C}, {Vp, C}, false);
    wpe.initialise("positional-embedding", {maxT, C}, false);

    ln1w.initialise("ln1w", {L, C}, false);
    ln1b.initialise("ln1b" , {L, C}, false);

    qkvw.initialise("qkv-weights" , {L, 3*C, C}, false);
    qkvb.initialise("qkv-bias" , {L, 3*C}, false);

    attprojw.initialise("attention-weights" , {L, C, C}, false);
    attprojb.initialise("attention-bias" , {L, C}, false);

    ln2w.initialise("ln2w" , {L, C}, false);
    ln2b.initialise("ln2b" , {L, C}, false);

    fcw.initialise("fcw" , {L, 4*C, C}, false);
    fcb.initialise("fcb" , {L, 4*C}, false);

    fcprojw.initialise("fcprojw" , {L, C, 4*C}, false);
    fcprojb.initialise("fcprojb" , {L, C}, false);

    lnfw.initialise("lnfw" , {C}, false);
    lnfb.initialise("lnfb" , {C}, false);
    */
    
    return true;
  }

  template<typename index_type, typename value_type>
  std::ofstream& operator<<(std::ofstream& ofs, gpt2_weights<index_type, value_type>& weights)
  {
    /*
    ofs.write((char*)&weights.V, sizeof(weights.V));
    ofs.write((char*)&weights.C, sizeof(weights.C));
    ofs.write((char*)&weights.L, sizeof(weights.L));
    ofs.write((char*)&weights.maxT, sizeof(weights.maxT));

    ofs << weights.wte;
    ofs << weights.wpe;

    ofs << weights.ln1w;
    ofs << weights.ln1b;

    ofs << weights.qkvw;
    ofs << weights.qkvb;

    ofs << weights.attprojw;
    ofs << weights.attprojb;

    ofs << weights.ln2w;
    ofs << weights.ln2b;

    ofs << weights.fcw;
    ofs << weights.fcb;

    ofs << weights.fcprojw;
    ofs << weights.fcprojb;

    ofs << weights.lnfw;
    ofs << weights.lnfb;
    */

    for(auto name:gpt2_weights<index_type, value_type>::names)
      {
	ofs << *(weights.at(name));
      }
    
    return ofs;
  }

  template<typename index_type, typename value_type>
  std::ifstream& operator>>(std::ifstream& ifs, gpt2_weights<index_type, value_type>& weights)
  {
    /*
    ifs.read((char*)&weights.V, sizeof(weights.V));
    ifs.read((char*)&weights.C, sizeof(weights.C));
    ifs.read((char*)&weights.L, sizeof(weights.L));
    ifs.read((char*)&weights.maxT, sizeof(weights.maxT));

    ifs >> weights.wte;
    ifs >> weights.wpe;

    ifs >> weights.ln1w;
    ifs >> weights.ln1b;

    ifs >> weights.qkvw;
    ifs >> weights.qkvb;

    ifs >> weights.attprojw;
    ifs >> weights.attprojb;

    ifs >> weights.ln2w;
    ifs >> weights.ln2b;

    ifs >> weights.fcw;
    ifs >> weights.fcb;

    ifs >> weights.fcprojw;
    ifs >> weights.fcprojb;

    ifs >> weights.lnfw;
    ifs >> weights.lnfb;
    */

    for(auto name:gpt2_weights<index_type, value_type>::names)
      {
	ifs >> *(weights.at(name));
      }
    
    return ifs;
  }

}

#endif
