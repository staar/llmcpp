//-*-C++-*-

#ifndef SHALLOW_TENSOR_H
#define SHALLOW_TENSOR_H

#include <string>
#include <vector>
#include <random>

namespace llmcpp
{
  template<typename index_type, typename value_type>
  class shallow_tensor: public base_tensor<index_type, value_type>
  {
    const inline static std::string UNKNOWN_NAME = "<unknowm>";

    typedef shallow_tensor<index_type, value_type> this_type;
    
  public:

    shallow_tensor(std::string name,
		   std::vector<index_type> dims,
		   std::vector<index_type> ldims,
		   bool col_major, value_type* weights);

    
    
  private:
    
    value_type* weights;
  };

  template<typename index_type, typename value_type>
  shallow_tensor<index_type, value_type>::shallow_tensor(std::string name,
							 std::vector<index_type> dims,
							 std::vector<index_type> ldims,
							 bool col_major, value_type* weights):
    base_tensor<index_type, value_type>(name, dims, ldims, col_major),
    weights(weights)
  {}
  
}

#endif
