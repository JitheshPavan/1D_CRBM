// File: rbm.cpp
#include "rbm.h"
#include <algorithm> 
#include <vector>

void Rbm::init_nn(const neuralnet_id& nn_id, const int& nsites,const int& start_pos_, const int& hidden_density)
{
  int input_dim= 2*nsites;
  int units= input_dim;
  hidden_units_= 4;
  num_sites_=input_dim;
  num_params_=  (hidden_units_ +( num_sites_+(hidden_units_*num_sites_)));
  VL.resize(num_sites_);
  HL.resize(hidden_units_);
  kernel_.resize(num_sites_, hidden_units_);
  std::cout << "RBM called"<<std::endl;
}

void Rbm::get_rbm_parameters(const RealVector& pvec)
{
  int pos=0;
  int n =pos;
    for (int i=0; i<num_sites_; ++i) {
      VL(i)=pvec[n++];
    }
    for (int j=0; j<hidden_units_ ; ++j) {
      HL(j)= pvec[n++];
    }
    for (int i=0; i<num_sites_; ++i) {
      for (int j=0; j<hidden_units_ ; ++j) {
        kernel_(i,j)= pvec[n++];
      }
    }

}

void Rbm::get_vlayer(RealVector& sigma, const ivector& row) const
{
  for(int i=0;i< num_sites_; ++i) {
    if(row(i) == 1) sigma(i) = 1;
    else sigma(i) = 0;
  }
}

void Rbm::compute_theta_table(const ivector& row) 
{
  RealVector sig;
  sig.resize(num_sites_);
  get_vlayer(sig,row);
  theta_= HL + (kernel_.transpose()* sig);
}

std::complex<double> Rbm::get_rbm_amplitudes(const ivector& row) const
{
  RealVector sig;
  sig.resize(num_sites_);
  get_vlayer(sig,row);
  double temp=1;
  for(int i=0;i<hidden_units_;++i){
    temp =temp * std::cosh(theta_[i]) ; 
  }
  return ((std::exp(VL.transpose() * sig))*temp);
}

double Rbm::sign_value(void) const
{

  std::complex<double> sign = std::cos(PI*theta_sign_);  
  return std::real(sign);
}

void Rbm::update_theta_table(const int& spin, const int& tsite, const int& fsite) const
{ 
  int ts,fs;
  int tot_sites_=num_sites_/2;
  int num_vunits_=num_sites_;
  if(spin==0){
    ts = tsite + tot_sites_;
    fs = fsite + tot_sites_;
  }
  else{
    ts = tsite;
    fs = fsite;
  }
  int k=0;
    for(int j=0; j<num_vunits_; j++){
      int d_ = k+j;
      double val = (kernel_(ts,d_) - kernel_(fs,d_));
      theta_(j) += val;
    }
    k += num_vunits_;
}

void Rbm::get_derivatives(const RealVector& pvec, ComplexVector& derivatives, const ivector& row, const int& start_pos) const
{
  RealVector input_;
  input_.resize(num_sites_);
  get_vlayer(input_, row);
  for(int i=0;i<num_sites_;++i){
    derivatives(i)=input_[i];
  }
  Vector grad_temp(hidden_units_);;  
  for(int i=0;i<hidden_units_;++i){
    grad_temp[i]=std::tanh(theta_[i]);
  }
  for(int i=0; i<hidden_units_; ++i){
    derivatives(num_sites_+i,0)= grad_temp[i];
  }
  Matrix grad_temp1 = input_* grad_temp.transpose();
  for(int i=0;i<num_sites_; ++i){
    for(int j=0; j<hidden_units_;++j){
      derivatives(num_sites_+hidden_units_+(hidden_units_*i)+j,0)= grad_temp1(i,j);
    }
  }
}


