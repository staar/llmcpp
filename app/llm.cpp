//-*-C++-*-

#include <set>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <ranges>

#include "libraries.h"

#include "utils.h"
#include "gpt2.h"

std::vector<std::string> get_modes()
{
  std::vector<std::string> modes
    = {
       "create-configs",
       "read-hf",
       "read",
       "write",
       "train-tokenizer",
       "apply-tokenizer",
       "train",
       "predict",
       "test"
  };
  return modes;
}

bool parse_arguments(int argc, char *argv[], nlohmann::json& args)
{
  //LOG_S(INFO) << __FUNCTION__;

  auto modes = get_modes();

  // Create a comma-separated string from the vector of strings
  std::string modes_
    = std::accumulate(modes.begin(), modes.end(), std::string(),
		      [](const std::string& a, const std::string& b) -> std::string
		      {
			return a.empty() ? b : a + ", " + b;
		      });

  cxxopts::Options options("llm", "llm");

  options.add_options()
    ("m,mode", "mode ["+modes_+"]",
     cxxopts::value<std::string>()->default_value("predict"))
    ("c,config", "configuration-file",
     cxxopts::value<std::string>()->default_value("null"))
    ("h,help", "print usage");

  auto result = options.parse(argc, argv);

  if(result.count("help")==1 or
     (result.count("help")==0 and result.count("mode")==0))
    {
      LOG_S(INFO) << options.help();
      return false;
    }

  if(result.count("mode")==0)
    {
      LOG_S(WARNING) << "`mode` is a required and needs to be one of "
                     << modes_;
      return false;
    }

  std::string mode = result["mode"].as<std::string>();
  args["mode"] = mode;

  if(std::find(modes.begin(), modes.end(), mode)==modes.end())
    {
      LOG_S(WARNING) << "mode `" << mode
                     << "` needs to be one of "
                     << modes_;
      return false;
    }

  if(mode=="create-configs" or mode=="test")
    {
      return true;
    }

  if(result.count("config")==0)
    {
      LOG_S(WARNING) << "`config` is required for `mode` of type "
                     << mode;
      return false;
    }

  if(result.count("mode") or
     result.count("config"))
    {
      std::string arg;

      arg = result["mode"].as<std::string>();
      args["mode"] = arg;

      arg = result["config"].as<std::string>();
      args["config"] = arg;

      arg = "no";
      if(result.count("interactive"))
        {
          arg = result["interactive"].as<std::string>();
        }

      if(arg=="yes" or arg=="y" or arg=="true")
        {
          args["interactive"] = true;
        }
      else
        {
          args["interactive"] = false;
        }

      LOG_S(INFO) << args.dump(2);
    }

  return true;
}

int update_args(nlohmann::json& args,
                nlohmann::json& config)
{
  LOG_S(INFO) << __FUNCTION__;

  if(args.is_null())
    {
      LOG_S(WARNING) << "args is null";
      return -1;
    }

  if(args.count("mode")==1 and
     args["mode"].get<std::string>()=="create-configs")
    {
      config["mode"]="create-configs";
      return 0;
    }

  if(args.count("config")==1 and
     args["config"]!="null")
    {
      std::string cfile = args["config"];

      std::ifstream ifs(cfile);
      if(ifs.good())
        {
          LOG_S(INFO) << "reading " << cfile;
          ifs >> config;
        }
      else
        {
          LOG_S(ERROR) << "can not read " << cfile;
          return -2;
        }
    }
  else
    {
      return -1;
    }

  // use the mode in the args to overwrite the config-mode
  if(args.count("mode")==1)
    {
      config["mode"] = args["mode"].get<std::string>();
      return 0;
    }

  return 0;
}

int test_llm()
{
  llmcpp::matmul<int, float>::test_forward();
  llmcpp::matmul<int, float>::test_backward();

  llmcpp::attention<int, float>::test_forward();
  llmcpp::attention<int, float>::test_backward();

  return 0;
}

template<typename model_type>
int create_configs(std::shared_ptr<model_type> model)
{
  auto config = model->create_config();
  LOG_S(WARNING) << "config: " << config.dump(2);

  std::ofstream ofs("config-example.json");
  if(ofs)
    {
      ofs << config;
    }

  return 0;
}

template<typename tokenizer_type>
int apply_tokenizer(std::shared_ptr<tokenizer_type> tokenizer)
{
  //
  tokenizer->load("../resources/tokenizers/gpt2/vocab.json",
		  "../resources/tokenizers/gpt2/merges.txt");

  auto tokens = tokenizer->encode("The old wise man was impressed with the sea [12-34].");

  for(auto token:tokens)
    {
      LOG_S(INFO) << "\t" << token << "\t" << tokenizer->decode(token);
    }

  return 1;
}

template<typename model_type>
int train_llm(nlohmann::json& config, std::shared_ptr<model_type> model)
{
  model->initialise(config);

  std::vector<std::vector<int> > inputs = {};
  std::vector<std::vector<int> > outputs = {};

  int B = model->get_B();
  int maxT = model->get_maxT();

  inputs.resize(B, {});
  outputs.resize(B, {});
  for(int b=0; b<B; b++)
    {
      for(int t=0; t<maxT; t++)
        {
          inputs.at(b).push_back(b+t);
          outputs.at(b).push_back(b+t+1);
        }
    }

  llmcpp::dense_tensor<int, int> itokens("itokens", {B, maxT}, false);
  llmcpp::dense_tensor<int, int> otokens("otokens", {B, maxT}, false);

  model->to_tensor(inputs, itokens);
  model->to_tensor(outputs, otokens);

  model->forward(itokens, otokens);
  model->backward(itokens, otokens);

  model->write("gpt2-model.bin");
  return 0;
}

template<typename model_type>
int predict_llm(nlohmann::json& config,
		std::shared_ptr<model_type> model)
{
  //auto mfile = config["model-file"].get<std::string>();

  model->initialise(config);
  
  //model->initialise_tokenizer();
  
  //model->read_hf_weights(mfile);

  auto tokenizer = model->get_tokenizer();
  
  /*
  llmcpp::gpt2_tokenizer tokenizer;
  tokenizer.load("../resources/tokenizers/gpt2/vocab.json",
                 "../resources/tokenizers/gpt2/merges.txt");
  */

  std::string prompt = "The old wise man was impressed with the";
  
  auto tokens = tokenizer->encode(prompt);
  //model->initialise(config);

  int B = model->get_B();
  int maxT = model->get_maxT();

  int topk = 10;      
  
  llmcpp::dense_tensor<int, int> itokens("itokens", {B, maxT}, false);
  llmcpp::dense_tensor<int, int> otokens("otokens", {B, 1}, false);

  llmcpp::dense_tensor<int, int> indices("indices", {B, maxT, topk}, false);
  llmcpp::dense_tensor<int, float> probs("probs", {B, maxT, topk}, false);
  
  std::vector<std::vector<int> > inputs(B, std::vector<int>({}));
  //std::vector<std::vector<int> > outputs(0);

  LOG_S(INFO) << "prompt: " << prompt; 
  for(auto token:tokens)
    {
      LOG_S(INFO) << std::setw(16) << token << std::setw(16) << tokenizer->decode(token);
      inputs.at(0).push_back(token);
    }

  for(int l=0; l<128; l++)
    {
      LOG_S(INFO) << "iteration: " << l;
      int seql = inputs.at(0).size();
      
      model->to_tensor(inputs, itokens);
      //model->to_tensor(outputs, otokens);
      
      model->forward(itokens, otokens);
      //model->backward(itokens, otokens);
      
      model->topk(seql, topk, indices, probs);

      int new_token = indices(0, seql-1, 0);
      LOG_S(WARNING) << std::setw(16) << new_token;
      LOG_S(WARNING) << std::setw(16) << tokenizer->decode(new_token);
      inputs.at(0).push_back(new_token);

      //std::string tmp;
      //std::cin >> tmp;

      //inputs.at(0).push_back(inputs.at(0).back());
    }
  
  /*
  for(int b=0; b<B; b++)
    {
      for(int t=0; t<maxT; t++)
	{
	  std::stringstream ss;
	  ss << std::setw(16) << itokens(b,t) << std::setw(16) << tokenizer->decode(itokens(b,t)) << " | ";
	  
	  for(int k=0; k<3; k++)
	    {
	      ss << std::setw(16) << probs(b,t,k) << std::setw(16) << tokenizer->decode(indices(b,t,k)) << " | ";
	    }

	  std::cout << ss.str() << "\n";
	}
    }
  */
  
  //model->write("gpt2-model.bin");

  return 1;
}

int main(int argc, char *argv[])
{
  loguru::init(argc, argv);

  nlohmann::json args;
  if(not parse_arguments(argc, argv, args))
    {
      return -1;
    }

  nlohmann::json config = nlohmann::json::object({});

  int res = update_args(args, config);
  if(res!=0) { LOG_S(WARNING) << "exiting ... "; return res; }

  LOG_S(INFO) << "config: \n" << config.dump(2);
  std::string mode = config.value("mode", "null");

  auto model = std::make_shared<llmcpp::gpt2_model<int, float> >();
  auto tokenizer = model->get_tokenizer();
  //llmcpp::gpt2_tokenizer tokenizer;
  
  if(mode=="test")
    {
      return test_llm();
    }
  else if(mode=="create-configs")
    {
      return create_configs(model);
    }
  else if(mode=="train-tokenizer")
    {
      return -1;
    }
  else if(mode=="apply-tokenizer")
    {
      apply_tokenizer(tokenizer);
    }
  else if(mode=="train")
    {
      return train_llm(config, model);

      /*
        model->initialise(config);

        std::vector<std::vector<int> > inputs = {};
        std::vector<std::vector<int> > outputs = {};

        int B = model->get_B();
        int maxT = model->get_maxT();

        inputs.resize(B, {});
        outputs.resize(B, {});
        for(int b=0; b<B; b++)
        {
        for(int t=0; t<maxT; t++)
        {
        inputs.at(b).push_back(b+t);
        outputs.at(b).push_back(b+t+1);
        }
        }

        llmcpp::dense_tensor<int, int> itokens("itokens", {B, maxT}, false);
        llmcpp::dense_tensor<int, int> otokens("otokens", {B, maxT}, false);

        model->to_tensor(inputs, itokens);
        model->to_tensor(outputs, otokens);

        model->forward(itokens, otokens);
        model->backward(itokens, otokens);

        model->write("gpt2-model.bin");
      */
    }
  else if(mode=="write-read")
    {
      auto mfile = config["model-file"].get<std::string>();
      model->write(mfile);
      model->read(mfile);
    }
  else if(mode=="read")
    {
      auto mfile = config["model-file"].get<std::string>();
      model->read_hf_weights(mfile);
    }
  else if(mode=="predict")
    {
      predict_llm(config, model);
    }
  else
    {
      LOG_S(ERROR) << "undefined mode: " << mode;
    }

  LOG_S(INFO) << "done ...";

  return 0;
}
