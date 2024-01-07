/*---------------------------------------------------------------------------
* @Author: amedhi
* @Date:   2019-03-19 13:12:20
* @Last Modified by:   Amal Medhi, amedhi@mbpro
* @Last Modified time: 2019-03-20 16:45:18
*----------------------------------------------------------------------------*/

#include <iostream>
#include <algorithm> 
#include <vector>
#include "vmc.h"
#include "lattice.h"
#include "basis.h"
#include "sysconfig.h"
#include "wavefunction.h"
#include "matrix.h"
//#include "rbm.h"
#include "CRBM.h"


//void get_visible_layer(ivector& sigma, const ivector& row);
int main(int argc, const char *argv[])
{
  VMC vmc;

  vmc.init();
  vmc.run_simulation();
}


