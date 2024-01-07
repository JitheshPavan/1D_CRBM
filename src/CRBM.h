#ifndef CRBM_H
#define CRBM_H

#include <complex>
#include <Eigen/Eigenvalues>
#include "wavefunction.h"
#include "./constants.h"
#include "./matrix.h"
#include "./lattice.h"
enum class neuralnet_id {RBM, FFN};
class CRbm{
public:
	CRbm() {};
	~CRbm() {};
	CRbm(const neuralnet_id& nn_id, const int& nsites, const int& start_pos_, const int& hidden_density)
  		{ init_nn( nn_id, nsites, start_pos_, hidden_density); }
	void init_nn(const neuralnet_id& nn_id, const int& nsites,const int& start_pos_, const int& hidden_density);
	void get_rbm_parameters(const RealVector& pvec);
	void get_vlayer(RealVector& sigma, const ivector& row) const;
	void compute_theta_table(const ivector& row);
	std::complex<double> get_rbm_amplitudes(const ivector& row) const;
	void get_derivatives(const RealVector& pvec, ComplexVector& derivatives, const ivector& row, const int& start_pos)const ;
	double sign_value(void) const;
	void update_theta_table(const int& spin, const int& tsite, const int& fsite) const;
	void output(const Vector& row)const;
	const int& num_vparams(void) const { return num_params_; }

private:
int num_sites_;
int kernel_size_;
int num_params_;
double param_b_;
Vector kernel_;
mutable Vector input_;
mutable Vector theta_;
};
#endif 
