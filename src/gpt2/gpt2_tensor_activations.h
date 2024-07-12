//-*-C++-*-

#ifndef GPT2_TENSOR_ACTIVATIONS_H
#define GPT2_TENSOR_ACTIVATIONS_H

#include <utils/llm_tensor.h>

namespace llmcpp
{
  template<typename index_type, typename value_type>
  class gpt2_activations
  {
    typedef dense_tensor<index_type, value_type> llm_tensor_type;

    const static inline std::vector<std::string> names =
      {
       "encoded",
       "ln1", "ln1_mean", "ln1_rstd",
       "qkv", "atty", "preatt", "att", "attproj",
       "residual2",
       "ln2", "ln2_mean", "ln2_rstd",
       "fch", "fch_gelu", "fcproj",
       "residual3",       
       "lnf", "lnf_mean", "lnf_rstd",
       "logits", "probs", "losses"
      };
    
  public:

    gpt2_activations() {}

    gpt2_activations(index_type L, // #-layers
                     index_type B, // batch-size
                     index_type T, // time-dimensions or sequence-length
                     index_type C, // #-channels or internal dimensions
                     index_type NH, // #-heads
                     index_type V); // vocabulary size

    bool initialise(gpt2_model_config& config);

    std::shared_ptr<llm_tensor_type> at(std::string name, bool strict=true);

    value_type* ptr(std::string name);
    value_type* ptr(std::string name, index_type layer);
    
    //gpt2_activations(const nlohmann::json& config);

    friend std::ofstream& operator<<(std::ofstream& ofs, gpt2_activations<index_type, value_type>& acts);
    friend std::ifstream& operator>>(std::ifstream& ifs, gpt2_activations<index_type, value_type>& acts);

  public:

    index_type L, B, T, C, NH, V, Vp;

    /*
    llm_tensor_type encoded; // (B, T, C)

    llm_tensor_type ln1; // (L, B, T, C)
    llm_tensor_type ln1_mean; // (L, B, T)
    llm_tensor_type ln1_rstd; // (L, B, T)
    llm_tensor_type qkv; // (L, B, T, 3*C)
    llm_tensor_type atty; // (L, B, T, C)
    llm_tensor_type preatt; // (L, B, NH, T, T)
    llm_tensor_type att; // (L, B, NH, T, T)
    llm_tensor_type attproj; // (L, B, T, C)
    llm_tensor_type residual2; // (L, B, T, C)
    llm_tensor_type ln2; // (L, B, T, C)
    llm_tensor_type ln2_mean; // (L, B, T)
    llm_tensor_type ln2_rstd; // (L, B, T)
    llm_tensor_type fch; // (L, B, T, 4*C)
    llm_tensor_type fch_gelu; // (L, B, T, 4*C)
    llm_tensor_type fcproj; // (L, B, T, C)
    llm_tensor_type residual3; // (L, B, T, C)

    llm_tensor_type lnf; // (B, T, C)
    llm_tensor_type lnf_mean; // (B, T)
    llm_tensor_type lnf_rstd; // (B, T)
    llm_tensor_type logits; // (B, T, V)
    llm_tensor_type probs; // (B, T, V)
    llm_tensor_type losses; // (B, T)
    */
    
    std::map<std::string, std::shared_ptr<llm_tensor_type> > tensors;
  };

  template<typename index_type, typename value_type>
  gpt2_activations<index_type, value_type>::gpt2_activations(index_type L,
                                                             index_type B,
                                                             index_type T,
                                                             index_type C,
                                                             index_type NH,
                                                             index_type V):
    L(L), B(B), T(T), C(C), NH(NH), V(V)//,

    /*
    encoded("encoded", {B, T, C}),
    ln1("layer_norm_1", {L, B, T, C}),
    ln1_mean("layer_norm_1_mean", {L, B, T}),
    ln1_rstd("layer_norm_1_rstd", {L, B, T}),
    qkv("qkv", {L, B, T, 3*C}),
    atty("atty", {L, B, T, C}),
    preatt("preattn", {L, B, NH, T, T}),
    att("att", {L, B, NH, T, T}),
    attproj("attproj", {L, B, T, C}),
    residual2("residual2", {L, B, T, C}),
    ln2("layer_norm_2", {L, B, T, C}),
    ln2_mean("layer_norm_2_mean", {L, B, T}),
    ln2_rstd("layer_norm_2_rstd", {L, B, T}),
    fch("fch", {L, B, T, 4*C}),
    fch_gelu("fch_gelu", {L, B, T, 4*C}),
    fcproj("fcproj", {L, B, T, C}),
    residual3("residual3", {L, B, T, C}),
    lnf("layer_norm_f", {B, T, C}),
    lnf_mean("layer_norm_f_mean", {B, T}),
    lnf_rstd("layer_norm_f_rstd", {B, T}),
    logits("logits", {B, T, V}),
    probs("probs", {B, T, V}),
    losses("losses", {B, T})
    */
  {}

  template<typename index_type, typename value_type>
  bool gpt2_activations<index_type, value_type>::initialise(gpt2_model_config& config)
  {
    LOG_S(INFO) << "initialising the GPT2 activations ...";
    
    B = config.batch_size;
    T = config.max_seq_len;
    L = config.num_layers;
    C = config.channels;
    NH = config.num_heads;
    V = config.vocab_size;
    Vp = config.padded_vocab_size;

    LOG_S(INFO) << B << "\t"
		<< T << "\t"
		<< L << "\t"
		<< C << "\t"
		<< NH << "\t"
		<< V << "\t"
		<< Vp;
    
    /*
    encoded.initialise("encoded", {B, T, C}, false);

    ln1.initialise("layer_norm_1", {L, B, T, C}, false);
    ln1_mean.initialise("layer_norm_1_mean", {L, B, T}, false);
    ln1_rstd.initialise("layer_norm_1_rstd", {L, B, T}, false);

    qkv.initialise("qkv", {L, B, T, 3*C}, false);
    atty.initialise("atty", {L, B, T, C}, false);
    preatt.initialise("preattn", {L, B, NH, T, T}, false);
    att.initialise("att", {L, B, NH, T, T}, false);
    attproj.initialise("attproj", {L, B, T, C}, false);

    residual2.initialise("residual2", {L, B, T, C}, false);

    ln2.initialise("layer_norm_2", {L, B, T, C}, false);
    ln2_mean.initialise("layer_norm_2_mean", {L, B, T}, false);
    ln2_rstd.initialise("layer_norm_2_rstd", {L, B, T}, false);
    
    fch.initialise("fch", {L, B, T, 4*C}, false);
    fch_gelu.initialise("fch_gelu", {L, B, T, 4*C}, false);
    fcproj.initialise("fcproj", {L, B, T, C}, false);

    residual3.initialise("residual3", {L, B, T, C}, false);

    lnf.initialise("layer_norm_f", {B, T, C}, false);
    lnf_mean.initialise("layer_norm_f_mean", {B, T}, false);
    lnf_rstd.initialise("layer_norm_f_rstd", {B, T}, false);

    logits.initialise("logits", {B, T, V}, {B, T, Vp}, false);
    probs.initialise("probs", {B, T, V}, {B, T, Vp}, false);

    losses.initialise("losses", {B, T}, false);
    */

    for(auto name:names)
      {
	tensors[name] = std::make_shared<llm_tensor_type>(name);
      }
    
    tensors.at("encoded")->initialise("encoded", {B, T, C}, false);
    
    tensors.at("ln1")->initialise("layer_norm_1", {L, B, T, C}, false);
    tensors.at("ln1_mean")->initialise("layer_norm_1_mean", {L, B, T}, false);
    tensors.at("ln1_rstd")->initialise("layer_norm_1_rstd", {L, B, T}, false);
    
    tensors.at("qkv")->initialise("qkv", {L, B, T, 3*C}, false);
    tensors.at("atty")->initialise("atty", {L, B, T, C}, false);
    tensors.at("preatt")->initialise("preattn", {L, B, NH, T, T}, false);
    tensors.at("att")->initialise("att", {L, B, NH, T, T}, false);
    tensors.at("attproj")->initialise("attproj", {L, B, T, C}, false);
    
    tensors.at("residual2")->initialise("residual2", {L, B, T, C}, false);
    
    tensors.at("ln2")->initialise("layer_norm_2", {L, B, T, C}, false);
    tensors.at("ln2_mean")->initialise("layer_norm_2_mean", {L, B, T}, false);
    tensors.at("ln2_rstd")->initialise("layer_norm_2_rstd", {L, B, T}, false);
    
    tensors.at("fch")->initialise("fch", {L, B, T, 4*C}, false);
    tensors.at("fch_gelu")->initialise("fch_gelu", {L, B, T, 4*C}, false);
    tensors.at("fcproj")->initialise("fcproj", {L, B, T, C}, false);
    
    tensors.at("residual3")->initialise("residual3", {L, B, T, C}, false);
    
    tensors.at("lnf")->initialise("layer_norm_f", {B, T, C}, false);
    tensors.at("lnf_mean")->initialise("layer_norm_f_mean", {B, T}, false);
    tensors.at("lnf_rstd")->initialise("layer_norm_f_rstd", {B, T}, false);
    
    tensors.at("logits")->initialise("logits", {B, T, V}, {B, T, Vp}, false);
    tensors.at("probs")->initialise("probs", {B, T, V}, {B, T, Vp}, false);
    
    tensors.at("losses")->initialise("losses", {B, T}, false);
    
    LOG_S(INFO) << "finalised GPT2 activations ...";
    
    return true;
  }

  template<typename index_type, typename value_type>
  std::shared_ptr<typename gpt2_activations<index_type, value_type>::llm_tensor_type> gpt2_activations<index_type, value_type>::at(std::string name, bool strict)
  {
    assert(find(names.begin(), names.end(), name)!=names.end());
    
    if((not strict) and tensors.count(name)==0)
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
  std::ofstream& operator<<(std::ofstream& ofs, gpt2_activations<index_type, value_type>& acts)
  {
    ofs.write((char*)&acts.L, sizeof(acts.L));
    ofs.write((char*)&acts.B, sizeof(acts.B));
    ofs.write((char*)&acts.T, sizeof(acts.T));
    ofs.write((char*)&acts.C, sizeof(acts.C));
    ofs.write((char*)&acts.NH, sizeof(acts.NH));
    ofs.write((char*)&acts.V, sizeof(acts.V));

    LOG_S(INFO) << "writing check-point ("
                << "L=" << acts.L << ", "
                << "B=" << acts.B << ", "
                << "T=" << acts.T << ", "
                << "C=" << acts.C << ", "
                << "NH=" << acts.NH << ", "
                << "V=" << acts.V << ")";

    for(auto name:gpt2_activations<index_type, value_type>::names)
      {
	ofs << *(acts.at(name));
      }
    
    /*
    ofs << acts.encoded;

    ofs << acts.ln1;
    ofs << acts.ln1_mean;
    ofs << acts.ln1_rstd;

    ofs << acts.qkv;
    ofs << acts.atty;
    ofs << acts.preattn;
    ofs << acts.att;
    ofs << acts.attproj;

    ofs << acts.residual2;

    ofs << acts.ln2;
    ofs << acts.ln2_mean;
    ofs << acts.ln2_rstd;

    ofs << acts.fch;
    ofs << acts.fch_gelu;

    ofs << acts.fcproj;

    ofs << acts.residual3;

    ofs << acts.lnf;
    ofs << acts.lnf_mean;
    ofs << acts.lnf_rstd;

    ofs << acts.logits;
    ofs << acts.probs;
    ofs << acts.losses;
    */
  }

  template<typename index_type, typename value_type>
  std::ifstream& operator>>(std::ifstream& ifs, gpt2_activations<index_type, value_type>& acts)
  {
    ifs.read((char*)&acts.L, sizeof(acts.L));
    ifs.read((char*)&acts.B, sizeof(acts.B));
    ifs.read((char*)&acts.T, sizeof(acts.T));
    ifs.read((char*)&acts.C, sizeof(acts.C));
    ifs.read((char*)&acts.NH, sizeof(acts.NH));
    ifs.read((char*)&acts.V, sizeof(acts.V));

    LOG_S(INFO) << "reading check-point ("
                << "L=" << acts.L << ", "
                << "B=" << acts.B << ", "
                << "T=" << acts.T << ", "
                << "C=" << acts.C << ", "
                << "NH=" << acts.NH << ", "
                << "V=" << acts.V << ")";

    for(auto name:gpt2_activations<index_type, value_type>::names)
      {
	ifs >> *(acts.at(name));
      }

    /*
    ifs >> acts.encoded;

    ifs >> acts.ln1;
    ifs >> acts.ln1_mean;
    ifs >> acts.ln1_rstd;

    ifs >> acts.qkv;
    ifs >> acts.atty;
    ifs >> acts.preattn;
    ifs >> acts.att;
    ifs >> acts.attproj;

    ifs >> acts.residual2;

    ifs >> acts.ln2;
    ifs >> acts.ln2_mean;
    ifs >> acts.ln2_rstd;

    ifs >> acts.fch;
    ifs >> acts.fch_gelu;

    ifs >> acts.fcproj;

    ifs >> acts.residual3;

    ifs >> acts.lnf;
    ifs >> acts.lnf_mean;
    ifs >> acts.lnf_rstd;

    ifs >> acts.logits;
    ifs >> acts.probs;
    ifs >> acts.losses;
    */
    
    return ifs;
  }

}

#endif
