//-*-C++-*-

#ifndef GPT2_TRAIN_CONFIG_H
#define GPT2_TRAIN_CONFIG_H

namespace llmcpp
{

  class gpt2_train_config
  {
  public:

    gpt2_train_config() {}

    nlohmann::json to_json();
    
  public:

  };

  nlohmann::json gpt2_train_config::to_json()
  {
    nlohmann::json config = nlohmann::json::object({});

    return config;
  }
  
}

#endif
