/*                                                                           
Developed by Sandeep Sharma and Garnet K.-L. Chan, 2012                      
Copyright (c) 2012, Garnet K.-L. Chan                                        
                                                                             
This program is integrated in Molpro with the permission of 
Sandeep Sharma and Garnet K.-L. Chan
*/

#include "IntegralMatrix.h"
#include <fstream>
#include "input.h"
#include "pario.h"
#include "global.h"
#include "orbstring.h"
#include "least_squares.h"
#include <include/communicate.h>
#include "sweepgenblock.h"
#include "npdm.h"

#ifdef _OPENMP
#include <omp.h>
#endif

//the following can be removed later
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/export.hpp>
#include <boost/format.hpp>
#include "spinblock.h"
#include "StateInfo.h"
#include "operatorfunctions.h"
#include "wavefunction.h"
#include "solver.h"
#include "davidson.h"
#include "guess_wavefunction.h"
#include "rotationmat.h"
#include "density.h"
#include "sweep.h"
#include "sweepCompress.h"
#include "BaseOperator.h"
#include "dmrg_wrapper.h"

#ifndef SERIAL
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "pario.h"


#ifdef USE_BTAS
void calculateOverlap();
#endif
void dmrg(double sweep_tol);
void compress(double sweep_tol, int targetState, int baseState);
void restart(double sweep_tol, bool reset_iter);
void dmrg_stateSpecific(double sweep_tol, int targetState);
void ReadInput(char* conf);
void fullrestartGenblock();
void license() {
#ifndef MOLPRO
  pout << "Block  Copyright (C) 2012  Garnet K.-L. Chan"<<endl;
  pout << "This program comes with ABSOLUTELY NO WARRANTY; for details see license file."<<endl;
  pout << "This is free software, and you are welcome to redistribute it"<<endl;
  pout << "under certain conditions; see license file for details."<<endl;
#endif
}


namespace SpinAdapted{
  Timer globaltimer(false);
  bool DEBUGWAIT = false;
  bool DEBUG_MEMORY = false;
  bool restartwarm = false;
  double NUMERICAL_ZERO = 1e-15;
  OneElectronArray v_1;
  TwoElectronArray v_2(TwoElectronArray::restrictedNonPermSymm);
  PairArray v_cc;
  CCCCArray v_cccc;
  CCCDArray v_cccd;
  Input dmrginp;
  int MAX_THRD = 1;
  bool FULLRESTART;
  bool RESTART;
  bool BACKWARD;
  bool reset_iter;
  std::vector<int> NPROP;
  int PROPBITLEN=1;
}

using namespace SpinAdapted;

int calldmrg(char* input, char* output)
{
  license();
  if (output != 0) {
    ofstream file;
    file.open(output);
    cout.rdbuf(file.rdbuf());
  }
  ReadInput(input);
  MAX_THRD = dmrginp.thrds_per_node()[mpigetrank()];
#ifdef _OPENMP
  omp_set_num_threads(MAX_THRD);
#endif

   //Initializing timer calls
  dmrginp.initCumulTimer();

  double sweep_tol = 1e-7;
  sweep_tol = dmrginp.get_sweep_tol();
  bool direction;
  int restartsize;
  SweepParams sweepParams;

  SweepParams sweep_copy;
  bool direction_copy; int restartsize_copy;
  Matrix O, H;




  switch(dmrginp.calc_type()) {

  case (COMPRESS):

    bool direction; int restartsize;
    sweepParams.restorestate(direction, restartsize);
    sweepParams.set_sweep_iter() = 0;
    restartsize = 0;
    //this genblock is required to generate all the nontranspose operators
    dmrginp.setimplicitTranspose() = false;
    SweepGenblock::do_one(sweepParams, false, false, false, restartsize, 0, 0);

    int targetState, baseState;
    targetState = 1; baseState = 0;
    compress(sweep_tol, targetState, baseState);
    break;
  case (CALCOVERLAP):
    pout.precision(12);
    if (mpigetrank() == 0) {
      for (int istate = 0; istate<dmrginp.nroots(); istate++) {
	bool direction;
	int restartsize;
	sweepParams.restorestate(direction, restartsize);
	Sweep::InitializeStateInfo(sweepParams, !direction, istate);
	Sweep::InitializeStateInfo(sweepParams, direction, istate);
	Sweep::CanonicalizeWavefunction(sweepParams, direction, istate);
	Sweep::CanonicalizeWavefunction(sweepParams, !direction, istate);
	Sweep::CanonicalizeWavefunction(sweepParams, direction, istate);
      }
      for (int istate = 0; istate<dmrginp.nroots(); istate++) 
	for (int j=istate; j<dmrginp.nroots() ; j++) {
	  Sweep::InitializeOverlapSpinBlocks(sweepParams, !direction, j, istate);
	  Sweep::InitializeOverlapSpinBlocks(sweepParams, direction, j, istate);
	}
      //Sweep::calculateAllOverlap(O);
    }
    break;

  case (CALCHAMILTONIAN):
    pout.precision(12);

    for (int istate = 0; istate<dmrginp.nroots(); istate++) {
      bool direction;
      int restartsize;
      sweepParams.restorestate(direction, restartsize);
      
      if (mpigetrank() == 0) {
	Sweep::InitializeStateInfo(sweepParams, !direction, istate);
	Sweep::InitializeStateInfo(sweepParams, direction, istate);
	Sweep::CanonicalizeWavefunction(sweepParams, !direction, istate);
	Sweep::CanonicalizeWavefunction(sweepParams, direction, istate);
	Sweep::CanonicalizeWavefunction(sweepParams, !direction, istate);
      }
    }
    
    //Sweep::calculateHMatrixElements(H);
    pout << "overlap "<<endl<<O<<endl;
    pout << "hamiltonian "<<endl<<H<<endl;
    break;

  case (DMRG):
    if (RESTART && !FULLRESTART)
      restart(sweep_tol, reset_iter);
    else if (FULLRESTART) {
      fullrestartGenblock();
      reset_iter = true;
      sweepParams.restorestate(direction, restartsize);
      sweepParams.calc_niter();
      sweepParams.savestate(direction, restartsize);
      restart(sweep_tol, reset_iter);
    }
    else if (BACKWARD) {
       fullrestartGenblock();
       reset_iter = true;
       sweepParams.restorestate(direction, restartsize);
       sweepParams.calc_niter();
       sweepParams.savestate(direction, restartsize);
       restart(sweep_tol, reset_iter);
    }
    else {
      dmrg(sweep_tol);
    }
    break;

  case (FCI):
    Sweep::fullci(sweep_tol);
    break;
    
  case (TINYCALC):
    Sweep::tiny(sweep_tol);
    break;
  case (ONEPDM):
    Npdm::npdm(1);
    break;

  case (TWOPDM):
    Npdm::npdm(2);
    break;

  case (THREEPDM):
    Npdm::npdm(3);
    break;

  case (FOURPDM):
    Npdm::npdm(4);
    break;

  case (NEVPT2PDM):
    Npdm::npdm(0);
    break;

  case (RESTART_ONEPDM):
    Npdm::npdm_restart(1);
    break;

  case (RESTART_TWOPDM):
    Npdm::npdm_restart(2);
    break;
  }

  return 0;
}


void calldmrg_(char* input, char* output) {
   int a;
   //a=calldmrg("dmrg.inp",0);//, output);
   a=calldmrg(input,0);//, output);
}


void fullrestartGenblock() {
  SweepParams sweepParams, sweepParamsTmp;
  bool direction; int restartsize;
  sweepParamsTmp.restorestate(direction, restartsize);
  sweepParams.set_sweep_iter() = 0;
  sweepParams.current_root() = -1;
  restartsize = 0;

  SweepGenblock::do_one(sweepParams, false, !direction, RESTART, restartsize, -1, -1);
  
  sweepParams.restorestate(direction, restartsize);
  sweepParams.set_sweep_iter()=0;
  sweepParams.set_block_iter() = 0;
  
  sweepParams.savestate(direction, restartsize);
}  


void restart(double sweep_tol, bool reset_iter)
{
  double last_fe = 100.;
  double last_be = 100.;
  double old_fe = 0.;
  double old_be = 0.;
  bool direction;
  int restartsize;
  SweepParams sweepParams;
  bool dodiis = false;

  int domoreIter = 2;

  sweepParams.restorestate(direction, restartsize);

  if (!dmrginp.setStateSpecific()) {
    if(reset_iter) { //this is when you restart from the start of the sweep
      sweepParams.set_sweep_iter() = 0;
      sweepParams.set_restart_iter() = 0;
    }
    
    if (restartwarm)
      last_fe = Sweep::do_one(sweepParams, true, direction, true, restartsize);
    else
      last_fe = Sweep::do_one(sweepParams, false, direction, true, restartsize);
    
    
    while ((fabs(last_fe - old_fe) > sweep_tol) || (fabs(last_be - old_be) > sweep_tol) || 
	   (dmrginp.algorithm_method() == TWODOT_TO_ONEDOT && dmrginp.twodot_to_onedot_iter()+1 >= sweepParams.get_sweep_iter()) )
      {
	
	old_fe = last_fe;
	old_be = last_be;
	if(dmrginp.max_iter() <= sweepParams.get_sweep_iter())
	  break;
	last_be = Sweep::do_one(sweepParams, false, !direction, false, 0);
	
	
	if(dmrginp.max_iter() <= sweepParams.get_sweep_iter())
	  break;
	last_fe = Sweep::do_one(sweepParams, false, direction, false, 0);	
      }
  }
  else { //this is state specific calculation  
    const int nroots = dmrginp.nroots();

    bool direction;
    int restartsize;
    sweepParams.restorestate(direction, restartsize);

    //initialize state and canonicalize all wavefunctions
    int currentRoot = sweepParams.current_root();
    for (int i=0; i<nroots; i++) {
      sweepParams.current_root() = i;
      if (mpigetrank()==0) {
	Sweep::InitializeStateInfo(sweepParams, direction, i);
	Sweep::InitializeStateInfo(sweepParams, !direction, i);
	Sweep::CanonicalizeWavefunction(sweepParams, direction, i);
	Sweep::CanonicalizeWavefunction(sweepParams, !direction, i);
	Sweep::CanonicalizeWavefunction(sweepParams, direction, i);
      }
    }

    //now generate overlaps with all the previous wavefunctions
    for (int i=0; i<currentRoot; i++) {
      sweepParams.current_root() = i;
      if (mpigetrank()==0) {
	for (int j=0; j<i; j++) {
	  Sweep::InitializeOverlapSpinBlocks(sweepParams, !direction, i, j);
	  Sweep::InitializeOverlapSpinBlocks(sweepParams, direction, i, j);
	}
      }
    }
    sweepParams.current_root() = currentRoot;

    if (dmrginp.outputlevel() > 0)
      if (sweepParams.current_root() <0) {
	pout << "This is most likely not a restart calculation and should be done without the restart command!!"<<endl;
	pout << "Aborting!!"<<endl;
	exit(0);
      }
      pout << "RESTARTING STATE SPECIFIC CALCULATION OF STATE "<<sweepParams.current_root()<<" AT SWEEP ITERATION  "<<sweepParams.get_sweep_iter()<<endl;

    //this is so that the iteration is not one ahead after generate block for restart
    --sweepParams.set_sweep_iter(); sweepParams.savestate(direction, restartsize);
    for (int i=sweepParams.current_root(); i<nroots; i++) {
      sweepParams.current_root() = i;

      if (dmrginp.outputlevel() > 0)
	pout << "RUNNING GENERATE BLOCKS FOR STATE "<<i<<endl;

      if (mpigetrank()==0) {
	Sweep::InitializeStateInfo(sweepParams, direction, i);
	Sweep::InitializeStateInfo(sweepParams, !direction, i);
	Sweep::CanonicalizeWavefunction(sweepParams, direction, i);
	Sweep::CanonicalizeWavefunction(sweepParams, !direction, i);
	Sweep::CanonicalizeWavefunction(sweepParams, direction, i);
	for (int j=0; j<i ; j++) {
	  Sweep::InitializeOverlapSpinBlocks(sweepParams, direction, i, j);
	  Sweep::InitializeOverlapSpinBlocks(sweepParams, !direction, i, j);
	}
      }
      SweepGenblock::do_one(sweepParams, false, !direction, false, 0, i, i);
      
      
      if (dmrginp.outputlevel() > 0)
	pout << "STATE SPECIFIC CALCULATION FOR STATE: "<<i<<endl;
      dmrg_stateSpecific(sweep_tol, i);
      if (dmrginp.outputlevel() > 0)
	pout << "STATE SPECIFIC CALCULATION FOR STATE: "<<i<<" FINSIHED"<<endl;

      sweepParams.set_sweep_iter() = 0;
      sweepParams.set_restart_iter() = 0;
      sweepParams.savestate(!direction, restartsize);
    }

    if (dmrginp.outputlevel() > 0)
      pout << "ALL STATE SPECIFIC CALCUALTIONS FINISHED"<<endl;
  }


  if(dmrginp.max_iter() <= sweepParams.get_sweep_iter()){
#ifndef MOLPRO
    pout << "Maximum sweep iterations achieved " << std::endl;
#else
    xout << "Maximum sweep iterations achieved " << std::endl;
#endif
  }

}

void dmrg(double sweep_tol)
{
  double last_fe = 10.e6;
  double last_be = 10.e6;
  double old_fe = 0.;
  double old_be = 0.;
  SweepParams sweepParams;

  int old_states=sweepParams.get_keep_states();
  int new_states;
  double old_error=0.0;
  double old_energy=0.0;
  // warm up sweep ...
  bool dodiis = false;

  int domoreIter = 0;
  bool direction;

  //this is regular dmrg calculation
  if(!dmrginp.setStateSpecific()) {
    sweepParams.current_root() = -1;
    last_fe = Sweep::do_one(sweepParams, true, true, false, 0);
    direction = false;
    while ((fabs(last_fe - old_fe) > sweep_tol) || (fabs(last_be - old_be) > sweep_tol) || 
	   (dmrginp.algorithm_method() == TWODOT_TO_ONEDOT && dmrginp.twodot_to_onedot_iter()+1 >= sweepParams.get_sweep_iter()) )
    {
      old_fe = last_fe;
      old_be = last_be;
      if(dmrginp.max_iter() <= sweepParams.get_sweep_iter())
	break;
      last_be = Sweep::do_one(sweepParams, false, false, false, 0);
      direction = true;
      if (dmrginp.outputlevel() > 0) 
	pout << "Finished Sweep Iteration "<<sweepParams.get_sweep_iter()<<endl;
      
      if(dmrginp.max_iter() <= sweepParams.get_sweep_iter())
	break;
      
      //For obtaining the extrapolated energy
      old_states=sweepParams.get_keep_states();
      new_states=sweepParams.get_keep_states_ls();
      
      last_fe = Sweep::do_one(sweepParams, false, true, false, 0);
      direction = false;
      
      new_states=sweepParams.get_keep_states();
      
      
      if (dmrginp.outputlevel() > 0)
	pout << "Finished Sweep Iteration "<<sweepParams.get_sweep_iter()<<endl;
      if (domoreIter == 2) {
	dodiis = true;
	break;
      }
      
    }
  }
  else { //this is state specific calculation  
    const int nroots = dmrginp.nroots();

    bool direction;
    int restartsize;
    //sweepParams.restorestate(direction, restartsize);
    //sweepParams.set_sweep_iter() = 0;
    //sweepParams.set_restart_iter() = 0;

    if (dmrginp.outputlevel() > 0)
      pout << "STARTING STATE SPECIFIC CALCULATION "<<endl;
    for (int i=0; i<nroots; i++) {
      sweepParams.current_root() = i;

      if (dmrginp.outputlevel() > 0)
	pout << "RUNNING GENERATE BLOCKS FOR STATE "<<i<<endl;

      if (mpigetrank()==0) {
	Sweep::InitializeStateInfo(sweepParams, direction, i);
	Sweep::InitializeStateInfo(sweepParams, !direction, i);
	Sweep::CanonicalizeWavefunction(sweepParams, direction, i);
	Sweep::CanonicalizeWavefunction(sweepParams, !direction, i);
	Sweep::CanonicalizeWavefunction(sweepParams, direction, i);
	Sweep::InitializeStateInfo(sweepParams, direction, i);
	Sweep::InitializeStateInfo(sweepParams, !direction, i);

      }

      for (int j=0; j<i ; j++) {
	Sweep::InitializeOverlapSpinBlocks(sweepParams, direction, i, j);
	Sweep::InitializeOverlapSpinBlocks(sweepParams, !direction, i, j);
      }

      SweepGenblock::do_one(sweepParams, false, !direction, false, 0, i, i);
      sweepParams.set_sweep_iter() = 0;
      sweepParams.set_restart_iter() = 0;
      sweepParams.savestate(!direction, restartsize);

      
      if (dmrginp.outputlevel() > 0)
	pout << "STATE SPECIFIC CALCULATION FOR STATE: "<<i<<endl;
      dmrg_stateSpecific(sweep_tol, i);
      if (dmrginp.outputlevel() > 0)
	pout << "STATE SPECIFIC CALCULATION FOR STATE: "<<i<<" FINSIHED"<<endl;
    }

    if (dmrginp.outputlevel() > 0)
      pout << "ALL STATE SPECIFIC CALCUALTIONS FINISHED"<<endl;
  }
}

void compress(double sweep_tol, int targetState, int baseState)
{
  double last_fe = 10.e6;
  double last_be = 10.e6;
  double old_fe = 0.;
  double old_be = 0.;
  SweepParams sweepParams;

  int old_states=sweepParams.get_keep_states();
  int new_states;
  double old_error=0.0;
  double old_energy=0.0;
  // warm up sweep ...
  bool dodiis = false;

  int domoreIter = 0;
  bool direction;

  sweepParams.current_root() = -1;
  last_fe = SweepCompress::do_one(sweepParams, true, true, false, 0, targetState, baseState);
  direction = false;
  while ( true)
    {
      old_fe = last_fe;
      old_be = last_be;
      if(dmrginp.max_iter() <= sweepParams.get_sweep_iter())
	break;
      last_be = SweepCompress::do_one(sweepParams, false, false, false, 0, targetState, baseState);
      direction = true;
      if (dmrginp.outputlevel() > 0) 
	pout << "Finished Sweep Iteration "<<sweepParams.get_sweep_iter()<<endl;
      
      if(dmrginp.max_iter() <= sweepParams.get_sweep_iter())
	break;
      
      //For obtaining the extrapolated energy
      old_states=sweepParams.get_keep_states();
      new_states=sweepParams.get_keep_states_ls();
      
      last_fe = SweepCompress::do_one(sweepParams, false, true, false, 0, targetState, baseState);
      direction = false;
      
      new_states=sweepParams.get_keep_states();
      
      
      if (dmrginp.outputlevel() > 0)
	pout << "Finished Sweep Iteration "<<sweepParams.get_sweep_iter()<<endl;
      
    }
  
}


void dmrg_stateSpecific(double sweep_tol, int targetState)
{
  double last_fe = 10.e6;
  double last_be = 10.e6;
  double old_fe = 0.;
  double old_be = 0.;
  int ls_count=0;
  SweepParams sweepParams;
  int old_states=sweepParams.get_keep_states();
  int new_states;
  double old_error=0.0;
  double old_energy=0.0;
  // warm up sweep ...

  bool direction;
  int restartsize;
  sweepParams.restorestate(direction, restartsize);

  //initialize array of size m_maxiter or dmrginp.max_iter() for dw and energy
  sweepParams.current_root() = targetState;

  last_fe = Sweep::do_one(sweepParams, false, direction, true, restartsize);

  while ((fabs(last_fe - old_fe) > sweep_tol) || (fabs(last_be - old_be) > sweep_tol)  )
    {
      old_fe = last_fe;
      old_be = last_be;
      if(dmrginp.max_iter() <= sweepParams.get_sweep_iter()) 
	break;

      last_be = Sweep::do_one(sweepParams, false, !direction, false, 0);
      if (dmrginp.outputlevel() > 0) 
         pout << "Finished Sweep Iteration "<<sweepParams.get_sweep_iter()<<endl;

      if(dmrginp.max_iter() <= sweepParams.get_sweep_iter())
	break;


      last_fe = Sweep::do_one(sweepParams, false, direction, false, 0);

      new_states=sweepParams.get_keep_states();


      if (dmrginp.outputlevel() > 0)
         pout << "Finished Sweep Iteration "<<sweepParams.get_sweep_iter()<<endl;

    }
  pout << "Converged Energy  " << sweepParams.get_lowest_energy()[0]+dmrginp.get_coreenergy()<< std::endl;
  if(dmrginp.max_iter() <= sweepParams.get_sweep_iter()) {
    
    pout << "Maximum sweep iterations achieved " << std::endl;
  }

  //one has to canonicalize the wavefunction with atleast 3 sweeps, this is a quirk of the way 
  //we transform wavefunction
  if (mpigetrank()==0) {
    Sweep::InitializeStateInfo(sweepParams, !direction, targetState);
    Sweep::InitializeStateInfo(sweepParams, direction, targetState);
    Sweep::CanonicalizeWavefunction(sweepParams, !direction, targetState);
    Sweep::CanonicalizeWavefunction(sweepParams, direction, targetState);
    Sweep::CanonicalizeWavefunction(sweepParams, !direction, targetState);
    Sweep::InitializeStateInfo(sweepParams, !direction, targetState);
    Sweep::InitializeStateInfo(sweepParams, direction, targetState);
    
  }

}


