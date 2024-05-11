//-*-C++-*-

#include <set>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <filesystem>

#include "libraries.h"

#include "utils.h"
#include "gpt2.h"

bool parse_arguments(int argc, char *argv[], nlohmann::json& args)
{
  LOG_S(INFO) << __FUNCTION__;
  
  cxxopts::Options options("llm", "llm");

  options.add_options()
    ("m,mode", "mode [create-configs,train,predict]",
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
		     << "`create-configs`, `train` or `predict`";
      return false;
    }

  std::string mode = result["mode"].as<std::string>();
  args["mode"] = mode;
  
  std::set<std::string> modes = {"create-configs","train","predict"};
  if(modes.count(mode)==0)
    {
      LOG_S(WARNING) << "mode `" << mode
		     << "` needs to be one of "
		     << "`create-configs`, `train` or `predict`";
      return false;
    }
  
  if(mode=="create-configs")
    {
      return true;
    }
  
  if(result.count("config")==0)
    {
      LOG_S(WARNING) << "`config` is required for `mode` of type "
		     << "`train` or `predict`.";
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
  
  return 0;
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
  
  if(mode=="create-configs")
    {
      auto config = model->create_config();
      LOG_S(WARNING) << "config: " << config.dump(2);
    }
  else if(mode=="train-tokenize")
    {
    }
  else if(mode=="apply-tokenize")
    {
    }
  else if(mode=="train")
    {
      model->initialise(config);

      model->write("gpt2-model.bin");
    }
  else if(mode=="predict")
    {
    }
  else
    {
      LOG_S(ERROR) << "undefined mode: " << mode;
    }

  LOG_S(INFO) << "done ...";
  
  return 0;
}
