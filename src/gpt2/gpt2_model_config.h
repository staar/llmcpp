//-*-C++-*-

#ifndef GPT2_MODEL_CONFIG_H
#define GPT2_MODEL_CONFIG_H

namespace llmcpp
{

  class gpt2_model_config
  {
  public:

    gpt2_model_config(int max_seq_len=1024,
		      int vocab_size=50257,
		      int padded_vocab_size=50304,
		      int num_layers=12,
		      int num_heads=12,
		      int channels=768);

    nlohmann::json to_json();
    
  public:

    int max_seq_len; // max sequence length, e.g. 1024
    int vocab_size; // vocab size, e.g. 50257
    int padded_vocab_size; // padded to e.g. %128==0, 50304
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
  };

  gpt2_model_config::gpt2_model_config(int max_seq_len,
                                       int vocab_size,
                                       int padded_vocab_size,
                                       int num_layers,
                                       int num_heads,
                                       int channels):
    max_seq_len(max_seq_len),
    vocab_size(vocab_size),
    padded_vocab_size(padded_vocab_size),
    num_layers(num_layers),
    num_heads(num_heads),
    channels(channels)
  {}

  nlohmann::json gpt2_model_config::to_json()
  {
    nlohmann::json config = nlohmann::json::object({});

    config["max_seq_len"] = max_seq_len;
    config["vocab_size"] = vocab_size;
    config["padded_vocab_size"] = padded_vocab_size;

    config["num_layers"] = num_layers;
    config["num_heads"] = num_heads;
    config["channels"] = channels;

    return config;
  }
  
}

#endif
