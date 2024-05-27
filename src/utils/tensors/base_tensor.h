//-*-C++-*-

#ifndef BASE_TENSOR_H
#define BASE_TENSOR_H

#include <string>
#include <vector>
#include <random>

namespace llmcpp
{
  template<typename index_type, typename value_type>
  class base_tensor
  {
    const inline static std::string UNKNOWN_NAME = "<unknowm>";

    typedef base_tensor<index_type, value_type> this_type;
    
  public:

    base_tensor(std::string name,
		std::vector<index_type> dims,
		std::vector<index_type> ldims,
		bool col_major);

    index_type ndim() { return dims.size(); }
    index_type size(index_type d) { return dims.at(d); }
    index_type lsize(index_type d) { return ldims.at(d); }
    
  protected:

    std::string to_string();
    
    void compute_steps();
    
  private:

    std::string name;

    bool col_major;

    std::vector<index_type> dims;

    std::vector<index_type> ldims;
    std::vector<index_type> steps;
  };

  template<typename index_type, typename value_type>
  base_tensor<index_type, value_type>::base_tensor(std::string name,
						   std::vector<index_type> dims,
						   std::vector<index_type> ldims,
						   bool col_major):
    name(name),
    dims(dims),
    ldims(ldims),
    col_major(col_major)
  {
    compute_steps();
  }
  
  template<typename index_type, typename value_type>
  std::string base_tensor<index_type, value_type>::to_string()
  {
    std::size_t size=1;
    for(std::size_t j=0; j<ldims.size(); j++)
      {
        size *= ldims.at(j);
      }

    std::stringstream ss1;
    ss1 << "[" << dims.at(0);
    for(std::size_t j=1; j<dims.size(); j++)
      ss1 << " x " << dims.at(j);
    ss1 << "]";

    std::stringstream ss2;
    ss2 << "[" << steps.at(0);
    for(std::size_t j=1; j<steps.size(); j++)
      ss2 << " x " << steps.at(j);
    ss2 << "]";

    std::stringstream ss;
    ss << std::setw(9) << std::scientific << sizeof(value_type)*double(size+0.0)/1.e9 << " GB"
       << " for " << name << "(D=" << ldims.size() << "): " << ss1.str() << ", " << ss2.str();

    return ss.str();
  }
  
  template<typename index_type, typename value_type>
  void base_tensor<index_type, value_type>::compute_steps()
  {
    this->steps = std::vector<index_type>(dims.size(), 1);
    
    if(col_major)
      {
        for(std::size_t i=0; i<steps.size(); i++)
          {
            for(std::size_t j=0; j<i; j++)
              {
                steps.at(i) *= ldims.at(j);
              }
          }
      }
    else
      {
        for(std::size_t i=0; i<steps.size(); i++)
          {
            for(std::size_t j=i+1; j<ldims.size(); j++)
              {
                steps.at(i) *= ldims.at(j);
              }
          }
      }
  }
  
}

#endif
