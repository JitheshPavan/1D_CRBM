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
#include <cmath>
void SysConfig::init(const lattice_id& lid, const lattice_size& size, const wf_id& wid, const neuralnet_id& nid)
{
  lattice_.construct(lid,size);
  num_sites_ = lattice_.num_sites();
  basis_state_.init(num_sites_);
  hole_doping_ = param[0];
  U_ = param[4];
  hidden_density_ = param[6];
  start_pos = 0;
  wf_.init(wid, lattice_, hole_doping_);
  rb_.init_nn(nid, num_sites_,start_pos, hidden_density_);
  num_upspins_ = wf_.num_upspins();
  num_dnspins_ = wf_.num_dnspins();
  num_exchange_moves_ = std::min(num_upspins_, num_dnspins_);
  basis_state_.init_spins(num_upspins_,num_dnspins_);
  num_wf_params_ = rb_.num_vparams();
  psi_mat_.resize(num_upspins_,num_dnspins_);
  psi_inv_.resize(num_upspins_,num_dnspins_);
  psi_row_.resize(num_dnspins_);
  psi_col_.resize(num_upspins_);
  inv_row_.resize(num_upspins_);
  psi_grad.resize(num_upspins_,num_dnspins_);

//sign structure
  sign_value=1;
  full_sign_value=1;
  no_doublons_=0;
  num_total_vparams_ = num_wf_params_ +1;// lambda counted.
  vparams_.resize(num_total_vparams_);

}

int SysConfig::build(const RealVector& vparams)
{
  rb_.get_rbm_parameters(vparams);
  param_lambda=vparams[num_total_vparams_-1];
  return 0;
}

int SysConfig::init_state(void){
  basis_state_.set_random();
  int num_attempt = 0;
  rb_.compute_theta_table(basis_state_.state());
  ffn_psi_ = rb_.get_rbm_amplitudes(basis_state_.state());
  num_updates_ = 0;
  refresh_cycle_ = 100;
  num_proposed_moves_ = 0;
  num_accepted_moves_ = 0;
  sign_value=1;
  full_sign_value=1;
  no_doublons_=0;
  return 0;
}

int SysConfig::update_state(void)
{

  for (int n=0; n<num_exchange_moves_; ++n) do_spin_exchange();
  for (int n=0; n<num_upspins_; ++n) do_upspin_hop();
  for (int n=0; n<num_dnspins_; ++n) do_dnspin_hop();
    num_updates_++;
  if (num_updates_ % refresh_cycle_ == 0) {
    rb_.compute_theta_table(basis_state_.state());
    ffn_psi_ = rb_.get_rbm_amplitudes(basis_state_.state()) * full_sign_value ;
  }
  return 0;
}

int SysConfig::do_upspin_hop(void)
{
  if (basis_state_.gen_upspin_hop()) {
    int upspin = basis_state_.which_upspin();
    int to_site = basis_state_.which_site();
    int fr_site = basis_state_.from_which_site();
    doublon_increament_= basis_state_.is_doublon_created();
    if(doublon_increament_== 1){
      no_doublons_ +=1;
      double sign_value_temp=sign_structure(to_site,fr_site);
      sign_value *= sign_value_temp;
      full_sign_value *= sign_value_temp * phase_shift(1,to_site,fr_site,basis_state_.state());
    }else if(doublon_increament_ ==-1){
      no_doublons_ +=-1;
      double sign_value_temp=1/sign_structure(to_site,fr_site);
      sign_value *= sign_value_temp;
      full_sign_value *= sign_value_temp * phase_shift_inverse(1 ,to_site,fr_site,basis_state_.state()) ;
     }else if(doublon_increament_>1){
      throw std::range_error("SysConfig::do_upspin_hop");
    }
    rb_.update_theta_table(1,to_site,fr_site);
    amplitude_t psi = (rb_.get_rbm_amplitudes(basis_state_.state()) *full_sign_value) ;
    amplitude_t nn_ratio = psi/ffn_psi_;
    amplitude_t weight_ratio = nn_ratio;
    double transition_proby = std::norm(weight_ratio);
    num_proposed_moves_++;
    if (basis_state_.rng().random_real()<transition_proby) {
      num_accepted_moves_++;
      basis_state_.commit_last_move();
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
    if(doublon_increament_== 1){
      no_doublons_ +=1;
      double sign_value_temp=sign_structure(to_site,fr_site);
      sign_value *= sign_value_temp;
      full_sign_value *= sign_value_temp * phase_shift(0, to_site,fr_site,basis_state_.state());
     }else if(doublon_increament_ ==-1){
      double sign_value_temp=1/sign_structure(to_site,fr_site);
      no_doublons_ -=1;
      sign_value *= sign_value_temp;
      full_sign_value *= sign_value_temp * phase_shift_inverse(0, to_site,fr_site,basis_state_.state());
     }else if(doublon_increament_>1){
      throw std::range_error("SysConfig::do_upspin_hop");
    }
    rb_.update_theta_table(0,to_site,fr_site);
    amplitude_t psi = (rb_.get_rbm_amplitudes(basis_state_.state())*full_sign_value);
    amplitude_t nn_ratio = psi/ffn_psi_;
    amplitude_t weight_ratio = nn_ratio;
    double transition_proby = std::norm(weight_ratio);
    num_proposed_moves_++;
    if (basis_state_.rng().random_real()<transition_proby) {
      num_accepted_moves_++;
      basis_state_.commit_last_move();
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
    full_sign_value= basis_state_.marshall_sign();
    amplitude_t psi_ = rb_.get_rbm_amplitudes(basis_state_.state()) * full_sign_value;
    amplitude_t nn_ratio = psi_/ffn_psi_;
    amplitude_t weight_ratio = nn_ratio;
    double transition_proby = std::norm(weight_ratio);
    num_proposed_moves_++;
    if (basis_state_.rng().random_real()<transition_proby) {
      num_accepted_moves_++;
      basis_state_.commit_last_move();
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

  std::complex<double> sign_value_temp= full_sign_value;
  for (int i=0; i<lattice_.num_bonds(); ++i) {
    int src = lattice_.bond(i).src();
    int tgt = lattice_.bond(i).tgt();
    int phase = lattice_.bond(i).phase();
    if (basis_state_.op_cdagc_up(src,tgt)) {
      int upspin = basis_state_.which_upspin();
      int to_site = basis_state_.which_site();
      int fr_site = basis_state_.from_which_site();
      doublon_increament_= basis_state_.is_doublon_created();
      if(doublon_increament_== 1){
        sign_value_temp = (full_sign_value * sign_structure(to_site,fr_site)*phase_shift(1,to_site,fr_site,basis_state_.state()));
       /*
        std::cout << "upspin spin: \n";
       // std::cout << sign_structure(to_site,fr_site) <<"  " << phase_shift(1,to_site,fr_site,basis_state_.state())<< "\n";
        std::cout << phase_shift(1,to_site,fr_site,basis_state_.state())<< "  "<< basis_state_.marshall_sign()<<"\n";getchar();*/
      }else if(doublon_increament_ ==-1){
        sign_value_temp = (full_sign_value * (1/sign_structure(to_site,fr_site)) * phase_shift_inverse(1,to_site,fr_site, basis_state_.state())) ;
        /*
        std::cout << "up spin: \n";
        //std::cout << sign_structure(to_site,fr_site) <<"  " << phase_shift_inverse(1,to_site,fr_site,basis_state_.state())<<"\n";
        std::cout << phase_shift(1,to_site,fr_site,basis_state_.state())<< "  "<< basis_state_.marshall_sign()<<"\n";getchar();*/
      }else {
        sign_value_temp = (full_sign_value )/factorial(no_doublons_) ;

      }
      rb_.update_theta_table(1,to_site,fr_site);
      amplitude_t psi = rb_.get_rbm_amplitudes(basis_state_.state()) * sign_value_temp  ;
      rb_.update_theta_table(1,fr_site,to_site);
      amplitude_t nn_ratio = psi/ffn_psi_;
      bond_sum += std::real(double(basis_state_.op_sign())*(nn_ratio));
      //bond_sum += std::real(nn_ratio);
    }
    if (basis_state_.op_cdagc_dn(src,tgt)) {
      int dnspin = basis_state_.which_dnspin();
      int to_site = basis_state_.which_site();
      int fr_site = basis_state_.from_which_site();
      doublon_increament_ = basis_state_.is_doublon_created();
      if(doublon_increament_== 1){
        sign_value_temp = full_sign_value * sign_structure(to_site,fr_site)*phase_shift(0,to_site,fr_site,basis_state_.state());
        /*
        std::cout << "down spin: \n";
        std::cout << phase_shift(1,to_site,fr_site,basis_state_.state())<< "  "<< basis_state_.marshall_sign()<<"\n";getchar();

        //std::cout << sign_structure(to_site,fr_site) <<"  " << phase_shift(1,to_site,fr_site,basis_state_.state())<< "\n";
        */
      }else if(doublon_increament_ ==-1){
        
        sign_value_temp = full_sign_value * (1/sign_structure(to_site,fr_site)) * phase_shift_inverse(0,to_site,fr_site, basis_state_.state());
        /*
        std::cout << "down spin reverse: \n";
        //std::cout << sign_structure(to_site,fr_site) <<"  " << phase_shift_inverse(1,to_site,fr_site,basis_state_.state())<<"\n";
        std::cout << phase_shift(1,to_site,fr_site,basis_state_.state())<< "  "<< basis_state_.marshall_sign()<<"\n";getchar();*/
      }else {
        sign_value_temp = (full_sign_value )/factorial(no_doublons_) ;

      }
      rb_.update_theta_table(0,to_site,fr_site);
      amplitude_t psi = rb_.get_rbm_amplitudes(basis_state_.state()) * sign_value_temp   ;
      rb_.update_theta_table(0,fr_site,to_site);
      amplitude_t nn_ratio = psi/ffn_psi_;      
      bond_sum += std::real(double(basis_state_.op_sign())*(nn_ratio));
      //bond_sum += std::real((nn_ratio));
   //   bond_sum += std::real(double(basis_state_.op_sign())*(nn_ratio));
    // std::cout << psi <<" "<<ffn_psi_ << "  " << nn_ratio<< std::endl;getchar();
    }
  }
  double t=1.0;
  return ((-t*bond_sum)+(U_*U_sum))/num_sites_;
}

void SysConfig::params(void) const
{
  double  p_value;
  int i =0;
  char p[20];
  std::ifstream infile;
  infile.open("input.txt");
  if (!infile) {std::cout << "Unable to open file";}
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
  get_derivatives(derivatives);
  for (int i=0; i<num_total_vparams_; ++i){  
    grad_logpsi(i) = std::real(derivatives(i));
  }
 // std::cout << "derivatives:\n" <<derivatives<<std::endl;getchar();
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
double SysConfig::sign_structure(const int fr_site,const int to_site) const {
  return std::exp(-1* param_lambda * std::abs(to_site-fr_site)) ;
}
std::complex<double> SysConfig::phase_shift(const int spin , const int to_site,const int fr_site, const ivector& row)const{
  RealVector state;
  state.resize(2*num_sites_);
  get_vlayer(state,row);
  int ts,fs;
  int tot_sites_=num_sites_;
    if(spin==0){
    ts = to_site + tot_sites_;
    fs = fr_site + tot_sites_;
  }
  else{
    ts = to_site;
    fs = fr_site;
  }
  if(state[ts]==0){
    std::range_error("error");
  }
  state[ts]=0;
  state[fs]=1;

  double phase_shift2=0;
  for(int l=0;l<num_sites_;++l){
    if(l < to_site){
      phase_shift2 -=   (state(l) -state(num_sites_+l)-1);
    }if(l > to_site){
      phase_shift2 +=  (state(l) -state(num_sites_+l)-1);
    }
  }
  double phase_shift1=0;
  for(int l=0;l<num_sites_;++l){
    if(l < fr_site){
      phase_shift1 -=  (state(l) -state(num_sites_+l)-1);
    }if(l > fr_site){
      phase_shift1 += (state(l) -state(num_sites_+l)-1);
    }
  }
  std::complex<double> local_phase_shift(0 ,PI *(phase_shift2-phase_shift1)/2);
  return std::exp(local_phase_shift);
}

//spin=0 downspin spin=1 upspin value to_site and fr_site does not chnage
std::complex<double> SysConfig::phase_shift_inverse(const int spin, const int to_site_old,const int fr_site_old, const ivector& row)const{
  RealVector state;
  state.resize(2*num_sites_);
  get_vlayer(state,row);
  
  double fr_site=to_site_old;
  double to_site= fr_site_old;
  double phase_shift2=0;
  for(int l=0;l<num_sites_;++l){
    if(l < to_site){
      phase_shift2 -=  (state(l) -state(num_sites_+l)-1);
    }if(l > to_site){
      phase_shift2 +=  (state(l) -state(num_sites_+l)-1);
    }
  }
  double phase_shift1=0;
  for(int l=0;l<num_sites_;++l){
    if(l < fr_site){
      phase_shift1 -= (state(l) -state(num_sites_+l)-1);
    }if(l > fr_site){
      phase_shift1 +=  (state(l) -state(num_sites_+l)-1);
    }

  }
  std::complex<double> local_phase_shift(0,PI*(phase_shift2-phase_shift1)/2);
  return std::exp(local_phase_shift);
}
// for nearest neighbour hopping there will always will be -1 sign;. It is not the case when doublon is not created or destroyed. But at such cases we dont calculate 
// sign structures.
void SysConfig::get_vlayer(RealVector& sigma, const ivector& row) const
{
  for(int i=0;i< 2*num_sites_; ++i) {
    if(row(i) == 1) sigma(i) = 1;
    else sigma(i) = 0;
  }
}
void SysConfig::get_derivatives(ComplexVector& pvec)const {
  pvec(num_total_vparams_-1)= log(sign_value)/param_lambda;
}
double SysConfig::factorial(double num) const {
  if(num==0 || num== 1){
    return 1;
  }
  return num * factorial(num-1);
}