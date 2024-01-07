/*---------------------------------------------------------------------------
* @Author: Amal Medhi, amedhi@mbpro
* @Date:   2019-03-20 11:50:30
* @Last Modified by:   Amal Medhi, amedhi@mbpro
* @Last Modified time: 2019-03-20 11:51:26
*----------------------------------------------------------------------------*/
// File: sysconfig.h
#ifndef SYSCONFIG_H
#define SYSCONFIG_H

#include "lattice.h"
#include "wavefunction.h"
#include "basis.h"
//#include "rbm.h"
#include "CRBM.h"

using amplitude_t = std::complex<double>;

class SysConfig
{
public:
	SysConfig() {}
	SysConfig(const lattice_id& lid, const lattice_size& size, const wf_id& wid, const neuralnet_id& nid) 
		{ init(lid, size, wid, nid); }
	~SysConfig() {}
	void init(const lattice_id& lid, const lattice_size& size, const wf_id& wid, const neuralnet_id& nid);
	int build(const RealVector& vparams);
	int init_state(void);
	int update_state(void);
	const int& num_vparams(void) const { return num_total_vparams_; }
  void print_stats(std::ostream& os=std::cout) const;
  double get_energy(void) const;
  void get_gradlog_Psi(RealVector& grad_logpsi) const;
  double_Array measure_gradient(const double& config_energy,RealVector& grad_logpsi_vec)const;
  void finalize(const double& mean_energy,RealVector& mean_config_value_,RealVector& energy_grad) const;
  double_Array product_grad_log_psi(const RealVector& grad_log_psi)const;
  void SR_matrix( RealMatrix& sr_matrix,const double_Array& u_triangular_mean)const;
  void params(void) const;
  double Double_occupancy(void) const;
  double get_sign(void) const;
  mutable double param[12];
private:
	Lattice lattice_;
	FockBasis basis_state_;
	int num_sites_;
	int num_upspins_;
	int num_dnspins_;
	int start_pos;
	int num_exchange_moves_;
	double hole_doping_;
	double U_;
	int hidden_density_;
	amplitude_t ffn_psi_;
	Wavefunction wf_;
	CRbm rb_;
	ComplexMatrix psi_mat_;
	ComplexMatrix psi_inv_;
	// variational parameters
	int num_total_vparams_;
	int num_wf_params_;
	RealVector vparams_;
	mutable ComplexMatrix psi_grad;

	// work arrays
  mutable ColVector psi_row_;
  mutable RowVector psi_col_;
  mutable RowVector inv_row_;

	// update parameters_
  int num_updates_;
  int refresh_cycle_;
  int num_proposed_moves_;
  int num_accepted_moves_;

  int do_upspin_hop(void);
  int do_dnspin_hop(void);
  int do_spin_exchange(void);
  int inv_update_upspin(const int& upspin, const ColVector& psi_row, 
    const std::complex<double>& det_ratio);
  int inv_update_dnspin(const int& dnspin, const RowVector& psi_col, 
    const std::complex<double>& det_ratio);
};


#endif
