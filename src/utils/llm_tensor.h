//-*-C++-*-

#ifndef LLM_TENSOR_H
#define LLM_TENSOR_H

namespace llmcpp
{
  template<typename index_type, typename value_type>
  class llm_tensor
  {
    const inline static std::string UNKNOWN_NAME = "<unknowm>";

  public:

    llm_tensor();
    llm_tensor(std::string name);

    llm_tensor(std::vector<index_type> dims);
    llm_tensor(std::vector<index_type> dims, bool col_major);

    llm_tensor(std::string name, std::vector<index_type> dims);
    llm_tensor(std::string name, std::vector<index_type> dims, bool col_major);

    index_type size(index_type d);
    
    void to_zero();
    
    bool initialise(std::string name, std::vector<index_type> dims, bool col_major);
    bool initialise(std::string name,
		    std::vector<index_type> dims,
		    std::vector<index_type> ldims,
		    bool col_major);

    value_type& operator()(index_type i);
    value_type& operator()(index_type i, index_type j);
    value_type& operator()(index_type i, index_type j, index_type k);
    value_type& operator()(index_type i, index_type j, index_type k, index_type l);
    
    value_type* ptr();
    value_type* ptr(const std::vector<index_type>& coor);

    std::size_t num_parameters();

    friend std::ofstream& operator<<(std::ofstream& ofs, const llm_tensor<index_type, value_type>& tnsr)
    {
      uint64_t name_len = tnsr.name.size();
      uint64_t dims_len = tnsr.dims.size();
      uint64_t vals_len = tnsr.weights->size();

      LOG_S(INFO) << "writing " << tnsr.name
                  << " [memsize: " << std::scientific << double(vals_len+0.0)/1.e9 << " Gb]";

      ofs.write((char*)&name_len, sizeof(name_len));
      ofs.write((char*)&dims_len, sizeof(dims_len));
      ofs.write((char*)&vals_len, sizeof(vals_len));

      ofs.write((char*)tnsr.name.data(), name_len*sizeof(char));
      ofs.write((char*)tnsr.dims.data(), dims_len*sizeof(index_type));

      ofs.write((char*)tnsr.ldims.data(), dims_len*sizeof(index_type));
      ofs.write((char*)tnsr.steps.data(), dims_len*sizeof(index_type));

      ofs.write((char*)tnsr.weights->data(), vals_len*sizeof(value_type));

      return ofs;
    }

    friend std::ifstream& operator>>(std::ifstream& ifs, llm_tensor<index_type, value_type>& tnsr)
    {
      uint64_t name_len, dims_len, vals_len;

      ifs.read((char*)&name_len, sizeof(name_len));
      ifs.read((char*)&dims_len, sizeof(dims_len));
      ifs.read((char*)&vals_len, sizeof(vals_len));

      LOG_S(INFO) << "reading " << tnsr.name
                  << " [memsize: " << std::scientific << vals_len << "]";

      tnsr.name.resize(name_len);
      tnsr.dims.resize(dims_len);
      tnsr.weights->resize(vals_len);

      ifs.read((char*)tnsr.name.data(), name_len*sizeof(char));
      ifs.read((char*)tnsr.dims.data(), dims_len*sizeof(index_type));
      ifs.read((char*)tnsr.ldims.data(), dims_len*sizeof(index_type));
      ifs.read((char*)tnsr.steps.data(), dims_len*sizeof(index_type));
      ifs.read((char*)tnsr.vals.data(), vals_len*sizeof(value_type));

      return ifs;
    }

  private:

    std::string name;

    bool col_major;

    std::vector<index_type> dims;

    std::vector<index_type> ldims;
    std::vector<index_type> steps;

    std::shared_ptr<std::vector<value_type> > weights;
  };

  template<typename index_type, typename value_type>
  llm_tensor<index_type, value_type>::llm_tensor():
    name(UNKNOWN_NAME),
    col_major(false),
    dims({}),
    ldims({}),
    steps({}),
    weights(NULL)
  {}

  template<typename index_type, typename value_type>
  llm_tensor<index_type, value_type>::llm_tensor(std::vector<index_type> dims):
    name(UNKNOWN_NAME),
    col_major(false),
    dims({}),
    ldims({}),
    steps({}),
    weights(NULL)
  {
    initialise(UNKNOWN_NAME, dims, false);
  }

  template<typename index_type, typename value_type>
  llm_tensor<index_type, value_type>::llm_tensor(std::vector<index_type> dims, bool col_major):
    name(UNKNOWN_NAME),
    col_major(false),
    dims({}),
    ldims({}),
    steps({}),
    weights(NULL)
  {
    initialise(UNKNOWN_NAME, dims, col_major);
  }

  template<typename index_type, typename value_type>
  llm_tensor<index_type, value_type>::llm_tensor(std::string name, std::vector<index_type> dims, bool col_major):
    name("<unknown>"),
    col_major(false),
    dims({}),
    ldims({}),
    steps({}),
    weights(NULL)
  {
    initialise(name, dims, col_major);
  }

  template<typename index_type, typename value_type>
  index_type llm_tensor<index_type, value_type>::size(index_type d)
  {
    assert(d<dims.size());
    return dims.at(d);
  }
  
  template<typename index_type, typename value_type>
  void llm_tensor<index_type, value_type>::to_zero()
  {
    if(weights==NULL)
      {
	LOG_S(ERROR) << "applied `to_zero` on uninitialised tensor ...";
	return;
      }

    for(auto itr=weights->begin(); itr!=weights->end(); itr++)
      {
	*itr = 0.0;
      }
  }
  
  template<typename index_type, typename value_type>
  bool llm_tensor<index_type, value_type>::initialise(std::string name,
						      std::vector<index_type> dims,
						      bool col_major)
  {
    return initialise(name, dims, dims, col_major);
  }

  template<typename index_type, typename value_type>
  bool llm_tensor<index_type, value_type>::initialise(std::string name,
						      std::vector<index_type> dims,
						      std::vector<index_type> ldims,
						      bool col_major)
  {
    this->name = name;
    this->col_major = col_major;

    assert(dims.size()==ldims.size());
    this->dims = dims;
    this->ldims = ldims;

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
    
    LOG_S(INFO) << " => init " 
		<< std::setw(9) << std::scientific << double(size+0.0)/1.e9 << " GB"
		<< " for " << name << "(D=" << ldims.size() << "): " << ss1.str() << ", " << ss2.str();
    
    weights = std::make_shared<std::vector<value_type> >(size, 0);
    assert(weights->size()==size);

    return true;
  }

  template<typename index_type, typename value_type>
  value_type& llm_tensor<index_type, value_type>::operator()(index_type i)
  {
    assert(dims.size()==1);
    assert(0<=i and i<dims.at(0));

    return weights->at(i);
  }

  template<typename index_type, typename value_type>
  value_type& llm_tensor<index_type, value_type>::operator()(index_type i, index_type j)
  {
    assert(dims.size()==2);
    assert(0<=i and i<dims.at(0));
    assert(0<=j and j<dims.at(1));

    index_type ind = 0;
    ind += steps.at(0)*i;
    ind += steps.at(1)*j;

    return weights->at(ind);
  }

  template<typename index_type, typename value_type>
  value_type& llm_tensor<index_type, value_type>::operator()(index_type i, index_type j, index_type k)
  {
    assert(dims.size()==3);
    assert(0<=i and i<dims.at(0));
    assert(0<=j and j<dims.at(1));
    assert(0<=k and j<dims.at(2));

    index_type ind = 0;
    ind += steps.at(0)*i;
    ind += steps.at(1)*j;
    ind += steps.at(2)*k;

    return weights->at(ind);
  }
  
  template<typename index_type, typename value_type>
  value_type& llm_tensor<index_type, value_type>::operator()(index_type i, index_type j,
							     index_type k, index_type l)
  {
    assert(dims.size()==4);
    assert(0<=i and i<dims.at(0));
    assert(0<=j and j<dims.at(1));
    assert(0<=k and j<dims.at(2));
    assert(0<=l and j<dims.at(3));

    index_type ind = 0;
    ind += steps.at(0)*i;
    ind += steps.at(1)*j;
    ind += steps.at(2)*k;
    ind += steps.at(3)*l;

    return weights->at(ind);
  }
  
  template<typename index_type, typename value_type>
  value_type* llm_tensor<index_type, value_type>::ptr()
  {
    return weights->data();
  }

  template<typename index_type, typename value_type>
  value_type* llm_tensor<index_type, value_type>::ptr(const std::vector<index_type>& coor)
  {
    assert(coor.size()==steps.size());

    index_type offset=0;
    for(std::size_t l=0; l<steps.size(); l++)
      {
        offset += coor.at(l)*steps.at(l);
      }

    value_type* result = weights->data();

    return result+offset;
  }

  template<typename index_type, typename value_type>
  std::size_t llm_tensor<index_type, value_type>::num_parameters()
  {
    if(dims.size()==0)
      {
        return 0;
      }

    std::size_t size=1;
    for(auto dim:dims)
      {
        size *= dim;
      }

    return size;
  }

  /*
    template<typename index_type, typename value_type>
    std::ofstream& operator<<(std::ofstream& ofs, const llm_tensor<index_type, value_type>& tnsr)
    {
    uint64_t name_len = tnsr.name.size();
    uint64_t dims_len = tnsr.dims.size();
    uint64_t vals_len = tnsr.weights->size();

    LOG_S(INFO) << "writing " << tnsr.name
    << " [memsize: " << std::scientific << vals_len << "]";

    ofs.write((char*)&name_len, sizeof(name_len));
    ofs.write((char*)&dims_len, sizeof(dims_len));
    ofs.write((char*)&vals_len, sizeof(vals_len));

    ofs.write((char*)tnsr.name.data(), name_len*sizeof(char));
    ofs.write((char*)tnsr.dims.data(), dims_len*sizeof(index_type));

    ofs.write((char*)tnsr.ldims.data(), dims_len*sizeof(index_type));
    ofs.write((char*)tnsr.steps.data(), dims_len*sizeof(index_type));

    ofs.write((char*)tnsr.weights->data(), vals_len*sizeof(value_type));
    }

    template<typename index_type, typename value_type>
    std::ifstream& operator>>(std::ifstream& ifs, llm_tensor<index_type, value_type>& tnsr)
    {
    uint64_t name_len, dims_len, vals_len;

    ifs.read((char*)&name_len, sizeof(name_len));
    ifs.read((char*)&dims_len, sizeof(dims_len));
    ifs.read((char*)&vals_len, sizeof(vals_len));

    LOG_S(INFO) << "reading " << tnsr.name
    << " [memsize: " << std::scientific << vals_len << "]";

    tnsr.name.resize(name_len);
    tnsr.dims.resize(dims_len);
    tnsr.weights->resize(vals_len);

    ifs.read((char*)tnsr.name.data(), name_len*sizeof(char));
    ifs.read((char*)tnsr.dims.data(), dims_len*sizeof(index_type));
    ifs.read((char*)tnsr.ldims.data(), dims_len*sizeof(index_type));
    ifs.read((char*)tnsr.steps.data(), dims_len*sizeof(index_type));
    ifs.read((char*)tnsr.vals.data(), vals_len*sizeof(value_type));

    return ifs;
    }
  */

}

#endif
