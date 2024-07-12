//-*-C++-*-

#ifndef DENSE_TENSOR_H
#define DENSE_TENSOR_H

#include <string>
#include <vector>
#include <random>

namespace llmcpp
{
  template<typename index_type, typename value_type>
  class dense_tensor
  {
    const inline static std::string UNKNOWN_NAME = "<unknowm>";

    typedef dense_tensor<index_type, value_type> this_type;
    typedef shallow_tensor<index_type, value_type> shallow_type;
    
  public:

    dense_tensor();
    dense_tensor(std::string name);

    dense_tensor(std::vector<index_type> dims);
    dense_tensor(std::vector<index_type> dims, bool col_major);

    dense_tensor(std::string name, std::vector<index_type> dims);
    dense_tensor(std::string name, std::vector<index_type> dims, bool col_major);

    index_type ndim() { return dims.size(); }

    index_type size() { index_type res=1; for(auto dim:ldims) { res *= dim; } return res; }
   
    index_type size(index_type d) { return dims.at(d); }
    index_type lsize(index_type d) { return ldims.at(d); }

    bool update_size(index_type d, index_type ndim)
    {
      assert(d<dims.size());
      assert(ndim<ldims.at(d));
      dims.at(d) = ndim;
    }
    
    void to_zero();
    void to_rand();
    
    this_type& initialise(std::string name,
			  std::vector<index_type> dims,
			  bool col_major);

    this_type& initialise(std::string name,
			  std::vector<index_type> dims,
			  std::vector<index_type> ldims,
			  bool col_major);

    value_type& operator[](index_type i);
    value_type& operator()(index_type i);
    value_type& operator()(index_type i, index_type j);
    value_type& operator()(index_type i, index_type j, index_type k);
    value_type& operator()(index_type i, index_type j, index_type k, index_type l);

    std::shared_ptr<shallow_type>& to_shallow(index_type i);
    std::shared_ptr<shallow_type>& to_shallow(index_type i, index_type j);
    
    value_type max_diff(this_type& tnsr);
    value_type max_reldiff(this_type& tnsr);

    typename std::vector<value_type>::iterator begin();
    typename std::vector<value_type>::iterator end();
    
    value_type* ptr();
    value_type* ptr(const std::vector<index_type>& coor);

    std::size_t num_parameters();

    friend std::ofstream& operator<<(std::ofstream& ofs, const dense_tensor<index_type, value_type>& tnsr)
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

    friend std::ifstream& operator>>(std::ifstream& ifs, dense_tensor<index_type, value_type>& tnsr)
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
      ifs.read((char*)tnsr.weights->data(), vals_len*sizeof(value_type));

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
  dense_tensor<index_type, value_type>::dense_tensor():
    name(UNKNOWN_NAME),
    col_major(false),
    dims({}),
    ldims({}),
    steps({}),
    weights(NULL)
  {}

  template<typename index_type, typename value_type>
  dense_tensor<index_type, value_type>::dense_tensor(std::string name):
    name(name),
    col_major(false),
    dims({}),
    ldims({}),
    steps({}),
    weights(NULL)
  {}
  
  template<typename index_type, typename value_type>
  dense_tensor<index_type, value_type>::dense_tensor(std::vector<index_type> dims):
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
  dense_tensor<index_type, value_type>::dense_tensor(std::vector<index_type> dims, bool col_major):
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
  dense_tensor<index_type, value_type>::dense_tensor(std::string name, std::vector<index_type> dims, bool col_major):
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
  void dense_tensor<index_type, value_type>::to_zero()
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
  void dense_tensor<index_type, value_type>::to_rand()
  {
    if(weights==NULL)
      {
	LOG_S(ERROR) << "applied `to_rand` on uninitialised tensor ...";
	return;
      }

    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Generate float numbers between 0 and 1
    std::uniform_real_distribution<value_type> dist(0.0f, 1.0f); 
    for(auto itr=weights->begin(); itr!=weights->end(); itr++)
      {
	*itr = dist(gen);
      }
  }

  template<typename index_type, typename value_type>
  dense_tensor<index_type, value_type>& dense_tensor<index_type, value_type>::initialise(std::string name,
										     std::vector<index_type> dims,
										     bool col_major)
  {
    return initialise(name, dims, dims, col_major);
  }

  template<typename index_type, typename value_type>
  dense_tensor<index_type, value_type>& dense_tensor<index_type, value_type>::initialise(std::string name,
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

    return *this;
  }

  template<typename index_type, typename value_type>
  value_type& dense_tensor<index_type, value_type>::operator()(index_type i)
  {
    assert(dims.size()==1);
    assert(0<=i and i<dims.at(0));

    return weights->at(i);
  }

  template<typename index_type, typename value_type>
  value_type& dense_tensor<index_type, value_type>::operator()(index_type i, index_type j)
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
  value_type& dense_tensor<index_type, value_type>::operator()(index_type i, index_type j, index_type k)
  {
    assert(dims.size()==3);
    assert(0<=i and i<dims.at(0));
    assert(0<=j and j<dims.at(1));
    assert(0<=k and k<dims.at(2));

    index_type ind = 0;
    ind += steps.at(0)*i;
    ind += steps.at(1)*j;
    ind += steps.at(2)*k;

    return weights->at(ind);
  }
  
  template<typename index_type, typename value_type>
  value_type& dense_tensor<index_type, value_type>::operator()(index_type i, index_type j,
							       index_type k, index_type l)
  {
    assert(dims.size()==4);
    assert(0<=i and i<dims.at(0));
    assert(0<=j and j<dims.at(1));
    assert(0<=k and k<dims.at(2));
    assert(0<=l and l<dims.at(3));

    index_type ind = 0;
    ind += steps.at(0)*i;
    ind += steps.at(1)*j;
    ind += steps.at(2)*k;
    ind += steps.at(3)*l;

    return weights->at(ind);
  }

  template<typename index_type, typename value_type>
  std::shared_ptr<shallow_tensor<index_type, value_type>>& dense_tensor<index_type, value_type>::to_shallow(index_type i)
  {
    auto N = this->ndims();
    
    std::vector<index_type> shallow_dims={}, shallow_ldims={};
    value_type shallow_ptr = this->ptr(); 

    if(col_major)
      {
	for(int l=0; l<N-1; l++)
	  {
	    shallow_dims.push_back(dims.at(l));
	    shallow_ldims.push_back(ldims.at(l));
	  }

	shallow_ptr = shallow_ptr + i*steps.at(N-1);
      }
    else
      {
	for(int l=1; l<N; l++)
	  {
	    shallow_dims.push_back(dims.at(l));
	    shallow_ldims.push_back(ldims.at(l));
	  }

	shallow_ptr = shallow_ptr + i*steps.at(0);
      }
    
    return std::make_shared<shallow_type>(name, shallow_dims, shallow_ldims, col_major, shallow_ptr);
  }
  
  template<typename index_type, typename value_type>
  std::shared_ptr<shallow_tensor<index_type, value_type> >& dense_tensor<index_type, value_type>::to_shallow(index_type i, index_type j)
  {
    auto N = this->ndims();
    
    std::vector<index_type> shallow_dims={}, shallow_ldims={};
    value_type shallow_ptr = this->ptr(); 

    if(col_major)
      {
	for(int l=0; l<N-2; l++)
	  {
	    shallow_dims.push_back(dims.at(l));
	    shallow_ldims.push_back(ldims.at(l));
	  }

	shallow_ptr = shallow_ptr + i*steps.at(N-2) + j*steps.at(N-1);
      }
    else
      {
	for(int l=2; l<N; l++)
	  {
	    shallow_dims.push_back(dims.at(l));
	    shallow_ldims.push_back(ldims.at(l));
	  }
	
	shallow_ptr = shallow_ptr + i*steps.at(0) + j*steps.at(1);
      }
    
    return std::make_shared<shallow_type>(name, shallow_dims, shallow_ldims, col_major, shallow_ptr);
  }
  
  template<typename index_type, typename value_type>
  value_type dense_tensor<index_type, value_type>::max_diff(this_type& tnsr)
  {
    if(dims.size()!=tnsr.ndim())
      {
	LOG_S(FATAL) << "mismatching dimensions for tensors";
      }

    for(int d=0; d<dims.size(); d++)
      {
	if(dims.at(d)!=tnsr.size(d))
	  {
	    LOG_S(FATAL) << "mismatching dimensions for tensors";
	  }
      }
        
    value_type result = 0, delta=0;
    
    auto itr_j = tnsr.begin();
    for(auto itr_i=weights->begin(); itr_i!=weights->end() and itr_j!=tnsr.end(); itr_i++, itr_j++)
      {
	delta = std::abs(*itr_i-*itr_j);
	result = std::max(delta, result); 
      }

    return result;
  }

  template<typename index_type, typename value_type>
  value_type dense_tensor<index_type, value_type>::max_reldiff(this_type& tnsr)
  {
    if(dims.size()!=tnsr.ndim())
      {
	LOG_S(FATAL) << "mismatching dimensions for tensors";
      }

    for(int d=0; d<dims.size(); d++)
      {
	if(dims.at(d)!=tnsr.size(d))
	  {
	    LOG_S(FATAL) << "mismatching dimensions for tensors";
	  }
      }
    
    value_type result = 0, diff=0, value=0;
    
    auto itr_j = tnsr.begin();
    for(auto itr_i=weights->begin(); itr_i!=weights->end() and itr_j!=tnsr.end(); itr_i++, itr_j++)
      {
	value = std::max(std::abs(*itr_i), std::abs(*itr_j));
	diff = std::abs(*itr_i-*itr_j);

	if(value>1.0)
	  {
	    diff = diff/value;
	  }
	
	result = std::max(diff, result); 
      }

    return result;
  }

  template<typename index_type, typename value_type>
  typename std::vector<value_type>::iterator dense_tensor<index_type, value_type>::begin()
  {
    return weights->begin();
  }
  
  template<typename index_type, typename value_type>
  typename std::vector<value_type>::iterator dense_tensor<index_type, value_type>::end()
  {
    return weights->end();
  }
  
  template<typename index_type, typename value_type>
  value_type* dense_tensor<index_type, value_type>::ptr()
  {
    return weights->data();
  }

  template<typename index_type, typename value_type>
  value_type* dense_tensor<index_type, value_type>::ptr(const std::vector<index_type>& coor)
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
  std::size_t dense_tensor<index_type, value_type>::num_parameters()
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

}

#endif
