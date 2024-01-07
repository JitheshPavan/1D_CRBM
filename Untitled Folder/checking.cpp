#include <iostream>
#include <cmath>
#include <iomanip>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iterator>
#include <vector>
#include <functional>
#include <complex> 
#include <chrono>

using namespace std;
using namespace Eigen;

int num_sites_=16;

enum class wf_id { BCS,FERMISEA};
void set_particle_num(const wf_id& id,const double& hole_doping) 
{
  wf_id id_;
  double hole_doping_ = hole_doping;
  double band_filling_ = 1.0-hole_doping_;
  if (id_==wf_id::BCS) {
    int n = static_cast<int>(std::round(0.5*band_filling_*num_sites_));
    //cout<<n<<endl;
    if (n<0 || n>num_sites_) throw std::range_error("Wavefunction:: hole doping out-of-range");
    int num_upspins_ = n;
    cout<<num_upspins_<<endl;
    int num_dnspins_ = num_upspins_;
    int num_spins_ = num_upspins_ + num_dnspins_;
    band_filling_ = static_cast<double>(2*n)/num_sites_;
  }
  else{
    int n = static_cast<int>(std::round(band_filling_*num_sites_));
    //cout<<n<<endl;
    if (n<0 || n>2*num_sites_) throw std::range_error("Wavefunction:: hole doping out-of-range");
    int num_spins_ = n;
    int num_dnspins_ = num_spins_/2;
    int num_upspins_ = num_spins_ - num_dnspins_;
    band_filling_ = static_cast<double>(n)/num_sites_;
  }
  hole_doping_ = 1.0 - band_filling_;
}

int main(){
set_particle_num(wf_id::BCS,0.0);
return 0;
}
