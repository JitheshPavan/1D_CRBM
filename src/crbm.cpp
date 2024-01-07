#include "CRBM.h"
#include <algorithm> 
#include <vector>
// hidden_density= length of kernel_
void CRbm::init_nn(const neuralnet_id& nn_id, const int& nsites,const int& start_pos_, const int& hidden_density)
{
  int input_dim= 2*nsites;
  kernel_size_= hidden_density;
  num_sites_=input_dim;
  theta_.resize(num_sites_);
  input_.resize(num_sites_);
  num_params_=  kernel_size_+ 1;
  param_b_=0.1;
  kernel_.resize(kernel_size_);
}
void CRbm::get_rbm_parameters(const RealVector& pvec){
  int n=0;
  for(int i=0;i< kernel_size_;i++){
    kernel_[i]=pvec[n++];
  }
  param_b_= pvec[n];
  if(!(n+1==num_params_)){
    throw std::range_error("error Crbm::get_rbm_parameters");
  }
}
void CRbm::get_vlayer(RealVector& sigma, const ivector& row) const{
  for(int i=0;i< num_sites_; ++i) {
    if(row(i) == 1) sigma(i) = 1;
    else sigma(i) = 0;
  }
}

void CRbm::output(const Vector& row)const{
  theta_.setZero();
  for(int j=0;j<num_sites_;++j){
    for(int k=0;k<kernel_size_;++k){
      int temperal=k+j;
      while(temperal>=num_sites_){
        temperal -= num_sites_;
      }
      theta_[j]+=  kernel_[k]* input_[temperal];
    }
    theta_[j]+= param_b_;
  }
}

void CRbm::compute_theta_table(const ivector& row) 
{
  get_vlayer(input_,row);
  output(input_);
}
std::complex<double> CRbm::get_rbm_amplitudes(const ivector& row) const
{
  double temp=1;
  for(int i=0;i<num_sites_;i++){
    temp *= std::cosh(theta_[i]);
  }
  return temp; 
}
double CRbm::sign_value(void) const
{
  std::range_error("Rbm::sign value: not implemented");
  return 2.0;
}
void CRbm::get_derivatives(const RealVector& pvec, ComplexVector& derivatives, const ivector& row, const int& start_pos) const{
  input_.resize(num_sites_);
  get_vlayer(input_, row);
  Matrix temperal_matrix(kernel_size_+1,num_sites_);
  temperal_matrix.setZero();
  for(int i=0;i<kernel_size_;++i){
    int temperal=0;
    for(int j=0;j<num_sites_;++j){
      temperal=i+j;
      while(temperal>=num_sites_){
      temperal -= num_sites_;
      }  
    temperal_matrix(i,j)=input_(temperal);
    }
  }
  for(int i=0;i<num_sites_;++i){
    temperal_matrix(kernel_size_,i)=1;
  }
  Vector temperal_matrix1(num_sites_);
  for(int i=0; i< num_sites_;++i){
    temperal_matrix1[i]= std::tanh(theta_[i]);
  }
  Vector gradient_= temperal_matrix * temperal_matrix1;
  for(int i=0;i<num_params_;++i){
    derivatives(i)= gradient_(i);
  }

}
void CRbm::update_theta_table(const int& spin, const int& tsite, const int& fsite) const
{ 
  int ts,fs;
  int tot_sites_=num_sites_/2;
    if(spin==0){
    ts = tsite + tot_sites_;
    fs = fsite + tot_sites_;
  }
  else{
    ts = tsite;
    fs = fsite;
  }
  input_[fs]=0;
  input_[ts]=1;
  output(input_);

}
