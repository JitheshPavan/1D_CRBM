/*-----------------------------------------------------------------------------
* @Author: Amal Medhi, amedhi@mbpro
* @Date:   2019-03-20 11:50:30
* @Last Modified by:   Amal Medhi, amedhi@mbpro
* @Last Modified time: 2019-03-28 10:15:42
*----------------------------------------------------------------------------*/
// File: sysconfig.cpp
#include <iomanip>
#include "sysconfig.h"
#include <fstream>
#include <ctime>

void SysConfig::init(const lattice_id& lid, const lattice_size& size, const wf_id& wid, const neuralnet_id& nid)
{
  lattice_.construct(lid,size);
  // one body part of the wavefunction
  num_sites_ = lattice_.num_sites();
  basis_state_.init(num_sites_);
  hole_doping_ = param[0];
  U_ = param[4];
  hidden_density_ = 10;
  start_pos = 0;
  wf_.init(wid, lattice_, hole_doping_);
  rb_.init_nn(nid, num_sites_,start_pos, hidden_density_);
  num_upspins_ = wf_.num_upspins();
  num_dnspins_ = wf_.num_dnspins();
  num_exchange_moves_ = std::min(num_upspins_, num_dnspins_);
  basis_state_.init_spins(num_upspins_,num_dnspins_);

  // variational parameters
  //num_wf_params_ = wf_.num_vparams()+rb_.num_vparams();
  num_wf_params_ = rb_.num_vparams();
  num_total_vparams_ = num_wf_params_;
  vparams_.resize(num_total_vparams_);

  // work arrays
  psi_mat_.resize(num_upspins_,num_dnspins_);
  psi_inv_.resize(num_upspins_,num_dnspins_);
  psi_row_.resize(num_dnspins_);
  psi_col_.resize(num_upspins_);
  inv_row_.resize(num_upspins_);
  psi_grad.resize(num_upspins_,num_dnspins_);
}

int SysConfig::build(const RealVector& vparams)
{
  rb_.get_rbm_parameters(vparams);
  return 0;
}

int SysConfig::init_state(void)
{
  // try for a well condictioned amplitude matrix
  basis_state_.set_random();
  int num_attempt = 0;
  /*while (true) {
    wf_.get_amplitudes(psi_mat_,basis_state_.upspin_sites(), basis_state_.dnspin_sites());
    // reciprocal conditioning number
    Eigen::JacobiSVD<ComplexMatrix> svd(psi_mat_);
    // reciprocal cond. num = smallest eigenval/largest eigen val
    double rcond = svd.singularValues()(svd.singularValues().size()-1)/svd.singularValues()(0);
    if (std::isnan(rcond)) rcond = 0.0; 
    if (rcond>1.0E-15) break;
    //std::cout << "rcondition number = "<< rcond << "\n";
    // try new basis state
    basis_state_.set_random();
    if (++num_attempt > 1000) {
      throw std::underflow_error("*SysConfig::init: configuration wave function ill conditioned.");
    }
  }*/

  //std::cout << psi_mat_ << "\n"; getchar();
  rb_.compute_theta_table(basis_state_.state());
  ffn_psi_ = rb_.get_rbm_amplitudes(basis_state_.state());
  //psi_inv_ = psi_mat_.inverse();
  // reset run parameters
  num_updates_ = 0;
  refresh_cycle_ = 100;
  num_proposed_moves_ = 0;
  num_accepted_moves_ = 0;
  return 0;
}

int SysConfig::update_state(void)
{
  //do_upspin_hop();
  //std::cout<<ffn_psi_ <<std::endl;
  for (int n=0; n<num_upspins_; ++n) do_upspin_hop();
  for (int n=0; n<num_dnspins_; ++n) do_dnspin_hop();
  for (int n=0; n<num_exchange_moves_; ++n) do_spin_exchange();
  num_updates_++;
  if (num_updates_ % refresh_cycle_ == 0) {
    rb_.compute_theta_table(basis_state_.state());
    ffn_psi_ = rb_.get_rbm_amplitudes(basis_state_.state());
  }
  //std::cout << basis_state_ << "\n"; getchar();
  return 0;
}

int SysConfig::do_upspin_hop(void)
{
  if (basis_state_.gen_upspin_hop()) {
    int upspin = basis_state_.which_upspin();
    int to_site = basis_state_.which_site();
    int fr_site = basis_state_.from_which_site();
    rb_.update_theta_table(1,to_site,fr_site);
    amplitude_t psi = rb_.get_rbm_amplitudes(basis_state_.state());
    /*wf_.get_amplitudes(psi_row_, to_site, basis_state_.dnspin_sites());
    amplitude_t det_ratio = psi_row_.cwiseProduct(psi_inv_.col(upspin)).sum();*/
    amplitude_t nn_ratio = psi/ffn_psi_;
    /*if (std::abs(det_ratio) < 1.0E-12) {
      // for safety
      basis_state_.undo_last_move();
      return 0; 
    } */
    //auto weight_ratio = det_ratio*nn_ratio;
    amplitude_t weight_ratio = nn_ratio;
    double transition_proby = std::norm(weight_ratio);
    num_proposed_moves_++;
    if (basis_state_.rng().random_real()<transition_proby) {
      num_accepted_moves_++;
      // upddate state
      basis_state_.commit_last_move();
      // update amplitudes
      //inv_update_upspin(upspin,psi_row_,det_ratio);
      ffn_psi_ = psi;
    }
    else {
      basis_state_.undo_last_move();
      rb_.update_theta_table(1,fr_site,to_site);
    }
  } 
  return 0;
}

int SysConfig::do_dnspin_hop(void)
{
  if (basis_state_.gen_dnspin_hop()) {
    int dnspin = basis_state_.which_dnspin();
    int to_site = basis_state_.which_site();
    int fr_site = basis_state_.from_which_site();
    rb_.update_theta_table(0,to_site,fr_site);
    amplitude_t psi = rb_.get_rbm_amplitudes(basis_state_.state());
    //wf_.get_amplitudes(psi_col_, basis_state_.upspin_sites(), to_site);
    //amplitude_t det_ratio = psi_col_.cwiseProduct(psi_inv_.row(dnspin)).sum();
    amplitude_t nn_ratio = psi/ffn_psi_;
    /*if (std::abs(det_ratio) < 1.0E-12) { // for safety
      basis_state_.undo_last_move();
      return 0; 
    } */
    //amplitude_t weight_ratio = det_ratio*nn_ratio;
    amplitude_t weight_ratio = nn_ratio;
    double transition_proby = std::norm(weight_ratio);
    num_proposed_moves_++;
    if (basis_state_.rng().random_real()<transition_proby) {
      num_accepted_moves_++;
      // upddate state
      basis_state_.commit_last_move();
      // update amplitudes
      //inv_update_dnspin(dnspin,psi_col_,det_ratio);
      ffn_psi_ = psi;
    }
    else {
      basis_state_.undo_last_move();
      rb_.update_theta_table(0,fr_site,to_site);
    }
  } 
  return 0;
}

int SysConfig::do_spin_exchange(void) 
{
  if (basis_state_.gen_exchange_move()){
    int upspin,dnspin,up_to_site,dn_to_site,up_fr_site,dn_fr_site;
    std::tie(upspin, up_to_site)    = basis_state_.exchange_move_uppart();
    std::tie(dnspin, dn_to_site)    = basis_state_.exchange_move_dnpart(); 
    std::tie(up_fr_site,dn_fr_site) = basis_state_.exchange_move_frsite();
    rb_.update_theta_table(1,up_to_site,up_fr_site);
    rb_.update_theta_table(0,dn_to_site,dn_fr_site); 
    amplitude_t psi_ = rb_.get_rbm_amplitudes(basis_state_.state());
    amplitude_t nn_ratio = psi_/ffn_psi_;
    amplitude_t weight_ratio = nn_ratio;
    double transition_proby = std::norm(weight_ratio);
    num_proposed_moves_++;
    if (basis_state_.rng().random_real()<transition_proby) {
      num_accepted_moves_++;
      // upddate state
      basis_state_.commit_last_move();
      // update amplitudes
      //inv_update_dnspin(dnspin,psi_col_,det_ratio);
      ffn_psi_ = psi_;
    }
    else {
      basis_state_.undo_last_move();
      rb_.update_theta_table(0,dn_fr_site,dn_to_site);
      rb_.update_theta_table(1,up_fr_site,up_to_site);
    }    
  }
  return 0;
}
int SysConfig::inv_update_upspin(const int& upspin, const ColVector& psi_row, 
  const amplitude_t& det_ratio)
{
  psi_mat_.row(upspin) = psi_row;
  amplitude_t ratio_inv = amplitude_t(1.0)/det_ratio;
  for (int i=0; i<upspin; ++i) {
    amplitude_t beta = ratio_inv*psi_row.cwiseProduct(psi_inv_.col(i)).sum();
    psi_inv_.col(i) -= beta * psi_inv_.col(upspin);
  }
  for (int i=upspin+1; i<num_upspins_; ++i) {
    amplitude_t beta = ratio_inv*psi_row.cwiseProduct(psi_inv_.col(i)).sum();
    psi_inv_.col(i) -= beta * psi_inv_.col(upspin);
  }
  psi_inv_.col(upspin) *= ratio_inv;
  return 0;
}

int SysConfig::inv_update_dnspin(const int& dnspin, const RowVector& psi_col, 
  const amplitude_t& det_ratio)
{
  psi_mat_.col(dnspin) = psi_col;
  amplitude_t ratio_inv = amplitude_t(1.0)/det_ratio;
  for (int i=0; i<dnspin; ++i) {
    amplitude_t beta = ratio_inv*psi_col_.cwiseProduct(psi_inv_.row(i)).sum();
    psi_inv_.row(i) -= beta * psi_inv_.row(dnspin);
  }
  for (int i=dnspin+1; i<num_dnspins_; ++i) {
    amplitude_t beta = ratio_inv*psi_col_.cwiseProduct(psi_inv_.row(i)).sum();
    psi_inv_.row(i) -= beta * psi_inv_.row(dnspin);
  }
  psi_inv_.row(dnspin) *= ratio_inv;
  return 0;
}

void SysConfig::print_stats(std::ostream& os) const
{
  std::streamsize dp = std::cout.precision(); 
  double accept_ratio = 100.0*double(num_accepted_moves_)/(num_proposed_moves_);
  //os << "--------------------------------------\n";
  //os << " total mcsteps = " << num_updates_ <<"\n";
  os << std::fixed << std::showpoint << std::setprecision(1);
  os << " acceptance ratio = " << accept_ratio << " %\n";
  //os << "--------------------------------------\n";
  // restore defaults
  os << std::resetiosflags(std::ios_base::floatfield) << std::setprecision(dp);
}

double SysConfig::get_energy(void) const
{
  // hopping energy
  double bond_sum = 0.0; double U_sum=0;
  for(int i=0; i<num_sites_; i++){
   U_sum += basis_state_.op_ni_updn(i);
  }
  //std::cout << basis_state_.state().transpose() << std::endl; 
  for (int i=0; i<lattice_.num_bonds(); ++i) {
    int src = lattice_.bond(i).src();
    int tgt = lattice_.bond(i).tgt();
    int phase = lattice_.bond(i).phase();
    //U_sum += basis_state_.op_ni_updn(src);
    // upspin hop
    if (basis_state_.op_cdagc_up(src,tgt)) {
      int upspin = basis_state_.which_upspin();
      int to_site = basis_state_.which_site();
      int fr_site = basis_state_.from_which_site();
      rb_.update_theta_table(1,to_site,fr_site);
      amplitude_t psi = rb_.get_rbm_amplitudes(basis_state_.state());
      rb_.update_theta_table(1,fr_site,to_site);
      amplitude_t nn_ratio = psi/ffn_psi_;
    
      bond_sum += std::real(double(basis_state_.op_sign())*(nn_ratio));
    }
    // dnspin hop
    if (basis_state_.op_cdagc_dn(src,tgt)) {
      int dnspin = basis_state_.which_dnspin();
      int to_site = basis_state_.which_site();
      int fr_site = basis_state_.from_which_site();
      rb_.update_theta_table(0,to_site,fr_site);
      amplitude_t psi = rb_.get_rbm_amplitudes(basis_state_.state());
      rb_.update_theta_table(0,fr_site,to_site);
      amplitude_t nn_ratio = psi/ffn_psi_;      
      bond_sum += std::real(double(basis_state_.op_sign())*(nn_ratio));
//      std::cout << "  "<<psi << " "<< ffn_psi_;getchar();

    }
  }
  //U_sum=U_sum/2;
//  std::cout <<"b ,U_,U_sum"<<bond_sum <<"  "<<U_ <<"  "<<U_sum<<std::endl;getchar();
  double t=1.0;
  return (-t*bond_sum+U_*U_sum)/num_sites_;
}

double SysConfig::get_sign(void) const
{
  double sign = rb_.sign_value();
  return sign;
}

void SysConfig::params(void) const
{
  double  p_value;
  //param[5];
  int i =0;
  char p[20];
  std::ifstream infile;
  infile.open("input.txt");
    if (!infile) {
        std::cout << "Unable to open file";
    }
  while ( i<12){
    infile>>p>>p_value;
    param[i]=p_value;
    i++;
  }
  infile.close();
}

double SysConfig::Double_occupancy(void) const {
  double D_sum=0.0;
  for(int i=0; i<num_sites_; i++){
    D_sum += basis_state_.op_ni_updn(i);
  }
  return D_sum/num_sites_;
}

void SysConfig::get_gradlog_Psi(RealVector& grad_logpsi) const
{
  ComplexVector derivatives(num_total_vparams_);  
  rb_.get_derivatives(vparams_, derivatives,basis_state_.state(),start_pos);
  for (int i=0; i<num_total_vparams_; ++i){  
    grad_logpsi(i) = std::real(derivatives(i));
  }

}

double_Array SysConfig::measure_gradient(const double& config_energy,RealVector& grad_logpsi_vec)const
{
  int n = 0;
  double_Array config_value(2*num_total_vparams_);
  for (int i=0; i<num_total_vparams_; ++i) {
    config_value[n] = config_energy*grad_logpsi_vec(i);
    config_value[n+1] = grad_logpsi_vec(i);
    n += 2;
  }
  return config_value;
}

void SysConfig::finalize(const double& mean_energy,RealVector& mean_config_value_,RealVector& energy_grad) const
{
  unsigned n = 0;
  for (unsigned i=0; i<num_total_vparams_; ++i) {
    energy_grad(i) = (mean_energy*mean_config_value_[n+1]-mean_config_value_[n]);
    n += 2;
  }
}

double_Array SysConfig::product_grad_log_psi(const RealVector& grad_log_psi)const
{ 
  int n=num_total_vparams_+num_total_vparams_*(num_total_vparams_+1)/2;
  double_Array u_triangular(n);
  for(int i=0;i<num_total_vparams_;++i) u_triangular[i]=grad_log_psi[i];
  int k=num_total_vparams_;
  for(int i=0;i<num_total_vparams_;++i){
    double x=grad_log_psi[i];
    for(int j=i;j<num_total_vparams_;++j){
      double y=grad_log_psi[j];
      u_triangular[k]=x*y;
      ++k;
    }
  }
  return u_triangular;
}

void SysConfig::SR_matrix(RealMatrix& sr_matrix,const double_Array& u_triangular_mean)const
{
  int k=num_total_vparams_;
  for(int i=0;i<num_total_vparams_;++i){
    double x=u_triangular_mean[i];
    for(int j=i;j<num_total_vparams_;++j){
      double y=u_triangular_mean[j];
      sr_matrix(i,j)=(u_triangular_mean[k]-x*y)/num_sites_;
      sr_matrix(j,i)=sr_matrix(i,j);
      ++k;  
    }
 }
}


