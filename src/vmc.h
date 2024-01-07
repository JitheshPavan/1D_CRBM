/*---------------------------------------------------------------------------
* @Author: Amal Medhi, amedhi@mbpro
* @Date:   2019-03-20 13:07:50
* @Last Modified by:   Amal Medhi, amedhi@mbpro
* @Last Modified time: 2019-03-20 13:08:22
*----------------------------------------------------------------------------*/
// File: vmc.h
#ifndef VMC_H
#define VMC_H

#include <iostream>
#include "sysconfig.h"
#include "mcdata/mc_observable.h"
//#include "rbm.h"

class VMC
{
public:
	VMC() {}
	~VMC() {}
	int init(void);
	int run_simulation(void);
private:
	CRbm rb_;
	SysConfig config;
	RealVector vparams;
	int num_vparams;
        RealMatrix sr_matrix;
        RealMatrix I;
        RealVector grads;
        RealVector grad_log_psi;
        RealMatrix U_;
        RealVector lambda;	
	int num_samples;
	int warmup_steps;
	int interval;
	double HI;
	double LO;
	double range;
	double epsilon;
	double step;
	double tol;	
	double lambda_max;
	int n;
    int num;
    int iterations;

	// observables
	mcdata::MC_Observable docc;
	mcdata::MC_Observable energy;
    //mcdata::MC_Observable energy_grad;
    mcdata::MC_Observable u_triangular;
	mcdata::MC_Observable f_values;
	mcdata::MC_Observable sign_value_;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es_; 	
};



#endif
