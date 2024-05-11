//-*-C++-*-

#ifndef GPT2_TENSORS_H
#define GPT2_TENSORS_H

#include <utils/llm_tensor.h>

namespace llmcpp
{

  template<typename index_type, typename value_type>
  class gpt2_weights
  {
    typedef llm_tensor<index_type, value_type> llm_tensor_type;

  public:

    gpt2_weights() {}

    gpt2_weights(index_type V,
                 index_type C,
                 index_type L,
                 index_type maxT);

    gpt2_weights(const nlohmann::json& config);

    bool initialise(gpt2_model_config& config);

    friend std::ofstream& operator<<(std::ofstream& ofs, const gpt2_weights<index_type, value_type>& weights)
    {
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

      return ofs;
    }

    friend std::ifstream& operator>>(std::ifstream& ifs, gpt2_weights<index_type, value_type>& weights)
    {
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

      ifs >> weights.attnprojw;
      ifs >> weights.attnprojb;

      ifs >> weights.ln2w;
      ifs >> weights.ln2b;

      ifs >> weights.fcw;
      ifs >> weights.fcb;

      ifs >> weights.fcprojw;
      ifs >> weights.fcprojb;

      ifs >> weights.lnfw;
      ifs >> weights.lnfb;

      return ifs;
    }

  public:

    index_type V, C, L, maxT;

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
  };

  template<typename index_type, typename value_type>
  gpt2_weights<index_type, value_type>::gpt2_weights(index_type V,
                                                     index_type C,
                                                     index_type L,
                                                     index_type maxT):
    V(V), C(C), L(L), maxT(maxT),

    wte("weight-embedding", {V, C}),
    wpe("positional-embedding", {maxT, C}),
    ln1w("ln1w", {L, C}),
    ln1b("ln1b" , {L, C}),
    qkvw("qkv-weights" , {L, 3*C, C}),
    qkvb("qkv-bias" , {L, 3*C}),
    attprojw("attention-weights" , {L, C, C}),
    attprojb("attention-bias" , {L, C}),
    ln2w("ln2w" , {L, C}),
    ln2b("ln2b" , {L, C}),
    fcw("fcw" , {L, 4*C, C}),
    fcb("fcb" , {L, 4*C}),
    fcprojw("fcprojw" , {L, C, 4*C}),
    fcprojb("fcprojb" , {L, C}),
    lnfw("lnfw" , {C}),
    lnfb("lnfb" , {C})
  {}

  template<typename index_type, typename value_type>
  bool gpt2_weights<index_type, value_type>::initialise(gpt2_model_config& config)
  {
    V = config.vocab_size;
    C = config.channels;
    L = config.num_layers;
    maxT = config.max_seq_len;

    LOG_S(INFO) << "V = " << V << "; C = " << C << "; L = " << L << "; maxT = " << maxT;
    
    wte.initialise("weight-embedding", {V, C}, false);
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

    return true;
  }

  /*
    template<typename index_type, typename value_type>
    std::ofstream& operator<<(std::ofstream& ofs, const gpt2_weights<index_type, value_type>& weights)
    {
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

    ofs << weights.attnprojw;
    ofs << weights.attnprojb;

    ofs << weights.ln2w;
    ofs << weights.ln2b;

    ofs << weights.fcw;
    ofs << weights.fcb;

    ofs << weights.fcprojw;
    ofs << weights.fcprojb;

    ofs << weights.lnfw;
    ofs << weights.lnfb;

    return ofs;
    }

    std::ofstream& operator<<(std::ofstream& ofs, const gpt2_weights<int, float>& weights)
    {
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

    return ofs;
    }

    template<typename index_type, typename value_type>
    std::ifstream& operator>>(std::ifstream& ifs, gpt2_weights<index_type, value_type>& weights)
    {
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

    ifs >> weights.attnprojw;
    ifs >> weights.attnprojb;

    ifs >> weights.ln2w;
    ifs >> weights.ln2b;

    ifs >> weights.fcw;
    ifs >> weights.fcb;

    ifs >> weights.fcprojw;
    ifs >> weights.fcprojb;

    ifs >> weights.lnfw;
    ifs >> weights.lnfb;

    return ifs;
    }
  */

  template<typename index_type, typename value_type>
  class gpt2_activations
  {
    typedef llm_tensor<index_type, value_type> llm_tensor_type;

  public:

    gpt2_activations() {}

    gpt2_activations(index_type L, // #-layers
                     index_type B, // batch-size
                     index_type T, // time-dimensions or sequence-length
                     index_type C, // #-channels or internal dimensions
                     index_type NH, // #-heads
                     index_type V); // vocabulary size

    //gpt2_activations(const nlohmann::json& config);

    friend std::ofstream& operator<<(std::ofstream& ofs, gpt2_activations<index_type, value_type>& acts);
    friend std::ifstream& operator>>(std::ifstream& ifs, gpt2_activations<index_type, value_type>& acts);

  public:

    index_type L, B, T, C, NH, V;

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
  };

  template<typename index_type, typename value_type>
  gpt2_activations<index_type, value_type>::gpt2_activations(index_type L,
                                                             index_type B,
                                                             index_type T,
                                                             index_type C,
                                                             index_type NH,
                                                             index_type V):
    L(L), B(B), T(T), C(C), NH(NH), V(V),

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
  {}

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

    return ifs;
  }

}

#endif
