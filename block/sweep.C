/*                                                                           
Developed by Sandeep Sharma and Garnet K.-L. Chan, 2012                      
Copyright (c) 2012, Garnet K.-L. Chan                                        
                                                                             
This program is integrated in Molpro with the permission of 
Sandeep Sharma and Garnet K.-L. Chan
*/

#include "guess_wavefunction.h"
#include "sweep.h"
#include "global.h"
#include "solver.h"
#include "initblocks.h"
#include "MatrixBLAS.h"
#include <boost/format.hpp>
#ifndef SERIAL
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif
#include "rotationmat.h"
#include "density.h"
#include "pario.h"

using namespace boost;
using namespace std;


//these blocks contain only the overlap operators, so they are cheap
void SpinAdapted::Sweep::makeSystemEnvironmentBigOverlapBlocks(const std::vector<int>& systemSites, SpinBlock& systemDot, SpinBlock& environmentDot,
							       SpinBlock& system, SpinBlock& newSystem, SpinBlock& environment, SpinBlock& newEnvironment,
							       SpinBlock& big, SweepParams& sweepParams, const bool& dot_with_sys, const bool& useSlater,
							       int braState, int ketState)
{
  bool forward = (systemSites [0] == 0);
  
  SpinBlock::restore(forward, systemSites, system, braState, ketState);
  if (!sweepParams.get_onedot() || dot_with_sys) {
    newSystem.initialise_op_array(OVERLAP, false);
    newSystem.setstoragetype(DISTRIBUTED_STORAGE);
    newSystem.BuildSumBlock (NO_PARTICLE_SPIN_NUMBER_CONSTRAINT, system, systemDot);
  }

  if (!dot_with_sys && sweepParams.get_onedot()) 
    InitBlocks::InitNewOverlapEnvironmentBlock(environment, systemDot, newEnvironment, system , systemDot,
					       braState, ketState, sweepParams.get_sys_add(), sweepParams.get_env_add(), 
					       forward, sweepParams.get_onedot(), dot_with_sys);
  else
    InitBlocks::InitNewOverlapEnvironmentBlock(environment, environmentDot, newEnvironment, system , systemDot,
					       braState, ketState, sweepParams.get_sys_add(), sweepParams.get_env_add(), 
					       forward, sweepParams.get_onedot(), dot_with_sys);

  if (!dot_with_sys && sweepParams.get_onedot())
    InitBlocks::InitBigBlock(system, newEnvironment, big); 
  else
    InitBlocks::InitBigBlock(newSystem, newEnvironment, big); 
}


void SpinAdapted::Sweep::makeSystemEnvironmentBigBlocks(SpinBlock& system, SpinBlock& systemDot, SpinBlock& newSystem, 
							SpinBlock& environment, SpinBlock& environmentDot, SpinBlock& newEnvironment,
							SpinBlock& big, SweepParams& sweepParams, const bool& dot_with_sys, const bool& useSlater)
{
  bool forward = (system.get_sites() [0] == 0);
  bool haveNormOps = dot_with_sys, haveCompOps = true;
  system.addAdditionalCompOps();

  const int nexact = forward ? sweepParams.get_forward_starting_size() : sweepParams.get_backward_starting_size();
  if (!sweepParams.get_onedot() || dot_with_sys) 
    InitBlocks::InitNewSystemBlock(system, systemDot, newSystem, sweepParams.current_root(), sweepParams.current_root(), sweepParams.get_sys_add(), dmrginp.direct(), 
				   DISTRIBUTED_STORAGE, haveNormOps, haveCompOps);

  if (!dot_with_sys && sweepParams.get_onedot()) 
    environmentDot = systemDot;
  InitBlocks::InitNewEnvironmentBlock(environment, environmentDot, newEnvironment, system, systemDot, sweepParams.current_root(), sweepParams.current_root(),
				      sweepParams.get_sys_add(), sweepParams.get_env_add(), forward, dmrginp.direct(),
				      sweepParams.get_onedot(), nexact, useSlater, !haveNormOps, haveCompOps, dot_with_sys);
  


  newSystem.set_loopblock(false); newEnvironment.set_loopblock(false); environment.set_loopblock(false); newEnvironment.set_loopblock(false);
  if (dot_with_sys) newSystem.set_loopblock(true);
  else newEnvironment.set_loopblock(true);
  if (!dot_with_sys && sweepParams.get_onedot())
    InitBlocks::InitBigBlock(system, newEnvironment, big); 
  else
    InitBlocks::InitBigBlock(newSystem, newEnvironment, big); 
}

void SpinAdapted::Sweep::BlockAndDecimate (SweepParams &sweepParams, SpinBlock& system, SpinBlock& newSystem, const bool &useSlater, const bool& dot_with_sys)
{
  if (dmrginp.outputlevel() > 0) {
    mcheck("at the start of block and decimate");
    pout << "\t\t\t dot with system "<<dot_with_sys<<endl;
  }
  pout <<endl<< "\t\t\t Performing Blocking"<<endl;
  // figure out if we are going forward or backwards
  dmrginp.guessgenT -> start();

  bool forward = (system.get_sites() [0] == 0);
  SpinBlock systemDot, environmentDot;
  int systemDotStart, systemDotEnd, environmentDotStart, environmentDotEnd;
  int systemDotSize = sweepParams.get_sys_add() - 1;
  int environmentDotSize = sweepParams.get_env_add() - 1;
  if (forward)
  {
    systemDotStart = dmrginp.spinAdapted() ? *system.get_sites().rbegin () + 1 : (*system.get_sites().rbegin ())/2 + 1 ;
    systemDotEnd = systemDotStart + systemDotSize;
    environmentDotStart = systemDotEnd + 1;
    environmentDotEnd = environmentDotStart + environmentDotSize;
  }
  else
  {
    systemDotStart = dmrginp.spinAdapted() ? system.get_sites()[0] - 1 : (system.get_sites()[0])/2 - 1 ;
    systemDotEnd = systemDotStart - systemDotSize;
    environmentDotStart = systemDotEnd - 1;
    environmentDotEnd = environmentDotStart - environmentDotSize;
  }
  systemDot = SpinBlock(systemDotStart, systemDotEnd, true);
  environmentDot = SpinBlock(environmentDotStart, environmentDotEnd, true);
  SpinBlock environment, newEnvironment;
  SpinBlock big;  // new_sys = sys+sys_dot; new_env = env+env_dot; big = new_sys + new_env then renormalize to find new_sys(new)
  makeSystemEnvironmentBigBlocks(system, systemDot, newSystem, environment, environmentDot, newEnvironment, big, sweepParams, dot_with_sys, useSlater);


  //analyse_operator_distribution(big);
  dmrginp.guessgenT -> stop();
  dmrginp.multiplierT -> start();
  std::vector<Matrix> rotatematrix;

  if (dmrginp.outputlevel() > 0)
    mcheck(""); 
  if (dmrginp.outputlevel() == 0) {
    if (!dot_with_sys && sweepParams.get_onedot()) pout << "\t\t\t System  Block"<<system;    
    else pout << "\t\t\t System  Block"<<newSystem;
    pout << "\t\t\t Environment Block"<<newEnvironment<<endl;
    pout << "\t\t\t Solving wavefunction "<<endl;
  }

  std::vector<Wavefunction> lowerStates;

  if(sweepParams.current_root() >= 0 ) {
    int originalOutputlevel = dmrginp.outputlevel();
    dmrginp.setOutputlevel() = -1;
    lowerStates.resize(sweepParams.current_root());

    DiagonalMatrix e;
    for (int istate = 0; istate<sweepParams.current_root(); istate++) {
      guessWaveTypes guesstype = sweepParams.get_block_iter() == 0 ? TRANSPOSE : TRANSFORM;

      //now one needs to make |phi> = O|psi> so that the |phi> has the same dimensions as our target state
      SpinBlock overlapBig;
      SpinBlock overlapsystem, overlapenvironment, overlapnewsystem, overlapnewenvironment;
      makeSystemEnvironmentBigOverlapBlocks(system.get_sites(), systemDot, environmentDot,
					    overlapsystem, overlapenvironment, overlapnewsystem, overlapnewenvironment,
					    overlapBig, sweepParams, dot_with_sys, useSlater,
					    sweepParams.current_root(), istate);

      GuessWave::guess_wavefunctions(lowerStates[istate], e, overlapBig, guesstype, sweepParams.get_onedot(), istate, dot_with_sys, 0.0);

      if (mpigetrank() == 0) {
	Wavefunction temp; temp.initialise(dmrginp.effective_molecule_quantum_vec(), &big, sweepParams.get_onedot());
	temp.Clear();
	overlapBig.multiplyOverlap(lowerStates[istate], &temp, MAX_THRD);
	lowerStates[istate] = temp;
      }
      overlapsystem.clear(); overlapenvironment.clear(); overlapnewsystem.clear(); overlapnewenvironment.clear();
    }
    dmrginp.setOutputlevel() = originalOutputlevel;
  }

  newSystem.RenormaliseFrom (sweepParams.set_lowest_energy(), sweepParams.set_lowest_energy_spins(), sweepParams.set_lowest_error(), 
                             rotatematrix, sweepParams.get_keep_states(), 
                             sweepParams.get_keep_qstates(), sweepParams.get_davidson_tol(), big, sweepParams.get_guesstype(), sweepParams.get_noise(), 
                             sweepParams.get_additional_noise(), sweepParams.get_onedot(), system, systemDot, environment, 
			     dot_with_sys, useSlater, sweepParams.get_sweep_iter(), sweepParams.current_root(), lowerStates);

  if (dmrginp.outputlevel() > 0)
    mcheck("");
  environment.clear();
  newEnvironment.clear();

  pout <<"\t\t\t Performing Renormalization "<<endl;
  pout << "\t\t\t Total discarded weight "<<sweepParams.get_lowest_error()<<endl<<endl;

  dmrginp.multiplierT -> stop();
  dmrginp.operrotT -> start();
  newSystem.transform_operators(rotatematrix);
  dmrginp.operrotT -> stop();

  //save the updated overlap spinblock
  if( sweepParams.current_root() >= 0 ) {
    int originalOutputlevel = dmrginp.outputlevel();
    dmrginp.setOutputlevel() = -1;
    for (int istate = 0; istate<sweepParams.current_root(); istate++) {
      SpinBlock overlapBig;
      SpinBlock overlapsystem, overlapenvironment, overlapnewsystem, overlapnewenvironment;
      SpinBlock overlapsystemDot(systemDotStart, systemDotEnd, true);
      SpinBlock overlapenvironmentDot(environmentDotStart, environmentDotEnd, true);
      guessWaveTypes guesstype = sweepParams.get_block_iter() == 0 ? TRANSPOSE : TRANSFORM;
      
      DiagonalMatrix e;
      makeSystemEnvironmentBigOverlapBlocks(system.get_sites(), overlapsystemDot, overlapenvironmentDot,
					    overlapsystem, overlapnewsystem, overlapenvironment, overlapnewenvironment,
					    overlapBig, sweepParams, true, useSlater,
					    sweepParams.current_root(), istate);

      Wavefunction iwave;
      GuessWave::guess_wavefunctions(iwave, e, overlapBig, guesstype, sweepParams.get_onedot(), istate, true, 0.0);
      std::vector<Matrix> ketrotatematrix;
      DensityMatrix tracedMatrix;
      tracedMatrix.allocate(overlapnewsystem.get_ketStateInfo());
      operatorfunctions::MultiplyProduct(iwave, Transpose(const_cast<Wavefunction&> (iwave)), tracedMatrix, 1.0);
      int largeNumber = 1000000;
      if (!mpigetrank())
	double error = makeRotateMatrix(tracedMatrix, ketrotatematrix, largeNumber, sweepParams.get_keep_qstates());

#ifndef SERIAL
      mpi::communicator world;
      broadcast(world, ketrotatematrix, 0);
#endif

      iwave.SaveWavefunctionInfo (overlapBig.get_ketStateInfo(), overlapBig.get_leftBlock()->get_sites(), istate);
      SaveRotationMatrix (overlapnewsystem.get_sites(), ketrotatematrix, istate);

      overlapnewsystem.transform_operators(rotatematrix, ketrotatematrix);
      SpinBlock::store(forward, overlapnewsystem.get_sites(), overlapnewsystem, sweepParams.current_root(), istate);
      overlapsystem.clear(); overlapenvironment.clear(); overlapnewsystem.clear(); overlapnewenvironment.clear();
    }
    dmrginp.setOutputlevel() = originalOutputlevel;
  }


  if (dmrginp.outputlevel() > 0)
    mcheck("after rotation and transformation of block");

  if (dmrginp.outputlevel() > 0){
    pout << *dmrginp.guessgenT<<" "<<*dmrginp.multiplierT<<" "<<*dmrginp.operrotT<< "  "<<globaltimer.totalwalltime()<<" timer "<<endl;
    pout << *dmrginp.makeopsT<<" makeops "<<endl;
    pout << *dmrginp.datatransfer<<" datatransfer "<<endl;
    pout <<"oneindexopmult   twoindexopmult   Hc  couplingcoeff"<<endl;  
    pout << *dmrginp.oneelecT<<" "<<*dmrginp.twoelecT<<" "<<*dmrginp.hmultiply<<" "<<*dmrginp.couplingcoeff<<" hmult"<<endl;
    pout << *dmrginp.buildsumblock<<" "<<*dmrginp.buildblockops<<" build block"<<endl;
    pout << "addnoise  S_0_opxop  S_1_opxop   S_2_opxop"<<endl;
    pout << *dmrginp.addnoise<<" "<<*dmrginp.s0time<<" "<<*dmrginp.s1time<<" "<<*dmrginp.s2time<<endl;
  }

}

double SpinAdapted::Sweep::do_one(SweepParams &sweepParams, const bool &warmUp, const bool &forward, const bool &restart, const int &restartSize)
{

  SpinBlock system;
  const int nroots = dmrginp.nroots(sweepParams.get_sweep_iter());

  std::vector<double> finalEnergy(nroots,1.0e10);
  std::vector<double> finalEnergy_spins(nroots,0.);
  double finalError = 0.;
  if (restart) {
    finalEnergy = sweepParams.get_lowest_energy();
    finalEnergy_spins = sweepParams.get_lowest_energy();
    finalError = sweepParams.get_lowest_error();
  }

  sweepParams.set_sweep_parameters();
  // a new renormalisation sweep routine
  pout << endl;
  if (forward)
    pout << "\t\t\t Starting sweep "<< sweepParams.set_sweep_iter()<<" in forwards direction"<<endl;
  else
    pout << "\t\t\t Starting sweep "<< sweepParams.set_sweep_iter()<<" in backwards direction" << endl;
  pout << "\t\t\t ============================================================================ " << endl;

  InitBlocks::InitStartingBlock (system,forward, sweepParams.current_root(), sweepParams.current_root(), sweepParams.get_forward_starting_size(), sweepParams.get_backward_starting_size(), restartSize, restart, warmUp);
  if(!restart)
    sweepParams.set_block_iter() = 0;

 
  if (dmrginp.outputlevel() > 0)
    pout << "\t\t\t Starting block is :: " << endl << system << endl;

  SpinBlock::store (forward, system.get_sites(), system, sweepParams.current_root(), sweepParams.current_root()); // if restart, just restoring an existing block --
  sweepParams.savestate(forward, system.get_sites().size());
  bool dot_with_sys = true;
  vector<int> syssites = system.get_sites();

  if (restart)
  {
    if (forward && system.get_complementary_sites()[0] >= dmrginp.last_site()/2)
      dot_with_sys = false;
    if (!forward && system.get_sites()[0]-1 < dmrginp.last_site()/2)
      dot_with_sys = false;
  }
  if (dmrginp.outputlevel() > 0)
    mcheck("at the very start of sweep");  // just timer

  for (; sweepParams.get_block_iter() < sweepParams.get_n_iters(); ) // get_n_iters() returns the number of blocking iterations needed in one sweep
    {
      pout << "\t\t\t Block Iteration :: " << sweepParams.get_block_iter() << endl;
      pout << "\t\t\t ----------------------------" << endl;
      if (dmrginp.outputlevel() > 0) {
	    if (forward)
	      pout << "\t\t\t Current direction is :: Forwards " << endl;
	    else
	      pout << "\t\t\t Current direction is :: Backwards " << endl;
      }

      if (dmrginp.no_transform() || (sweepParams.get_sweep_iter()-sweepParams.get_restart_iter() == 0 && sweepParams.get_block_iter() == 0))
	      sweepParams.set_guesstype() = BASIC;
      else if (!warmUp && sweepParams.get_block_iter() != 0) 
  	    sweepParams.set_guesstype() = TRANSFORM;
      else if (!warmUp && sweepParams.get_block_iter() == 0 && 
                ((dmrginp.algorithm_method() == TWODOT_TO_ONEDOT && dmrginp.twodot_to_onedot_iter() != sweepParams.get_sweep_iter()) ||
                  dmrginp.algorithm_method() != TWODOT_TO_ONEDOT))
        sweepParams.set_guesstype() = TRANSPOSE;
      else
        sweepParams.set_guesstype() = BASIC;

      
      if (dmrginp.outputlevel() > 0)
         pout << "\t\t\t Blocking and Decimating " << endl;
	  
      SpinBlock newSystem; // new system after blocking and decimating

      //Need to substitute by:
      if (warmUp && (sym=="dinfh" || sym=="trans" || sym == "dinfh_abelian" || NonabelianSym || dmrginp.hamiltonian()==HEISENBERG))
         Startup(sweepParams, system, newSystem);
      else {
         if (sweepParams.set_sweep_iter() == 1 && sweepParams.get_block_iter() == 0)
           sweepParams.set_guesstype() = BASIC;
         if(sweepParams.set_sweep_iter() == 1 && sweepParams.get_largest_dw()<=NUMERICAL_ZERO)
           sweepParams.set_additional_noise() = dmrginp.get_twodot_noise();
         BlockAndDecimate (sweepParams, system, newSystem, warmUp, dot_with_sys);
      }
      
      //Need to substitute by?

      if (!(warmUp && (sym=="trans" || sym == "dinfh_abelian" || NonabelianSym || dmrginp.hamiltonian()==HEISENBERG))){
	for(int j=0;j<nroots;++j)
	{
	  int istate = dmrginp.setStateSpecific() ? sweepParams.current_root() : j;
	  
#ifndef MOLPRO
	  pout << "\t\t\t Total block energy for State [ " << istate << 
	    " ] with " << sweepParams.get_keep_states()<<" States :: " << sweepParams.get_lowest_energy()[j]+dmrginp.get_coreenergy() <<endl;              
#else 
	  //We might want to relax the output restrictions here, so it prints out with outputlevel=0
	  if (dmrginp.outputlevel() < 0) {
	    pout << "\t\t\t Total block energy for State [ " << istate << 
	      " ] with " << sweepParams.get_keep_states()<<" States :: " << fixed << setprecision(10) << sweepParams.get_lowest_energy()[j]+dmrginp.get_coreenergy() <<endl;              
	  }
#endif
	}
	
	//this criteria should work for state average or state specific because the lowest sweep energy is always the lowest of the average
	finalEnergy_spins = ( (std::accumulate(sweepParams.get_lowest_energy().begin(), sweepParams.get_lowest_energy().end(),0.0) < std::accumulate(finalEnergy.begin(), finalEnergy.end(),0.0)) ? sweepParams.get_lowest_energy_spins() : finalEnergy_spins);
	finalEnergy = ((std::accumulate(sweepParams.get_lowest_energy().begin(), sweepParams.get_lowest_energy().end(),0.0) < std::accumulate(finalEnergy.begin(), finalEnergy.end(),0.0)) ? sweepParams.get_lowest_energy() : finalEnergy);
	finalError = max(sweepParams.get_lowest_error(),finalError);
      }
      
      system = newSystem;
      if (dmrginp.outputlevel() > 0){
	    pout << system<<endl;
	    //system.printOperatorSummary();
      }
      
      //system size is going to be less than environment size
      if (forward && system.get_complementary_sites()[0] >= dmrginp.last_site()/2)
	    dot_with_sys = false;
      if (!forward && system.get_sites()[0]-1 < dmrginp.last_site()/2)
	    dot_with_sys = false;

      SpinBlock::store (forward, system.get_sites(), system, sweepParams.current_root(), sweepParams.current_root());	 	
      syssites = system.get_sites();
      if (dmrginp.outputlevel() > 0)
	pout << "\t\t\t saving state " << syssites.size() << endl;
      ++sweepParams.set_block_iter();
      
#ifndef SERIAL
      mpi::communicator world;
      world.barrier();
#endif
      sweepParams.savestate(forward, syssites.size());
      if (dmrginp.outputlevel() > 0)
         mcheck("at the end of sweep iteration");
    }

  for(int j=0;j<nroots;++j) {
    int istate = dmrginp.setStateSpecific() ? sweepParams.current_root() : j;
    pout << "\t\t\t Finished Sweep with " << sweepParams.get_keep_states() << " states and sweep energy for State [ " << istate 
	 << " ] with Spin [ " << dmrginp.molecule_quantum().get_s()  << " ] :: " << finalEnergy[j]+dmrginp.get_coreenergy() << endl;
  }

  pout << "\t\t\t Largest Error for Sweep with " << sweepParams.get_keep_states() << " states is " << finalError << endl;
  sweepParams.set_largest_dw() = finalError;

  for(int j=0;j<nroots;++j){
    int istate = dmrginp.setStateSpecific() ? sweepParams.current_root() : j;
    if (mpigetrank() == 0) {
#ifndef MOLPRO
      printf("\t\t\t M = %6i  state = %4i  Largest Discarded Weight = %8.3e  Sweep Energy = %20.10f \n",sweepParams.get_keep_states(), istate, finalError, finalEnergy[j]+dmrginp.get_coreenergy());
#else 
      //printf("\t\t\t M = %6i   Largest Discarded Weight = %8.3e  Sweep Energy = %20.10f \n",sweepParams.get_keep_states(), finalError, finalEnergy[j]+dmrginp.get_coreenergy());
      xout << "\t\t\t M = " <<  setw(6) << sweepParams.get_keep_states() ; 
      xout << "\t Largest Discarded Weight = " << scientific << setprecision(3) << finalError ;
      xout << "\t Sweep Energy = " << fixed << setprecision(10) << finalEnergy[j]+dmrginp.get_coreenergy() << endl;
#endif
    }
  }
  pout << "\t\t\t ============================================================================ " << endl;

  // update the static number of iterations

  ++sweepParams.set_sweep_iter();
  if (!(warmUp && (sym=="trans" || sym == "dinfh_abelian" || NonabelianSym || dmrginp.hamiltonian()==HEISENBERG))){
    if (!mpigetrank())
    {
      std::string efile;
      efile = str(boost::format("%s%s") % dmrginp.load_prefix() % "/dmrg.e" );

      //if state specific only write back the current energy to the dmrg.e file and leave the rest unchanged
      if (dmrginp.setStateSpecific() ) {
	sweepParams.set_lowest_energy().resize(dmrginp.nroots());
	sweepParams.set_lowest_energy()[sweepParams.current_root()] = sweepParams.get_lowest_energy()[0];
	FILE* fin = fopen(efile.c_str(), "rb");
	for(int j=0;j<dmrginp.nroots();++j) {
	  double e;
	  fread( &e, 1, sizeof(double), fin);
	  if (j != sweepParams.current_root())
	    sweepParams.set_lowest_energy()[j] = e - dmrginp.get_coreenergy();
	}
	fclose(fin);
      }

      FILE* f = fopen(efile.c_str(), "wb");      
      for(int j=0;j<dmrginp.nroots();++j) {
	double e = sweepParams.get_lowest_energy()[j]+dmrginp.get_coreenergy(); //instead of the lowest energy of the sweep, we record the last energy of the sweep
	fwrite( &e, 1, sizeof(double), f);
      }
      fclose(f);
    }
  }

  return std::accumulate(finalEnergy.begin(), finalEnergy.end(),0.0)/dmrginp.nroots(sweepParams.get_sweep_iter());
}

void SpinAdapted::Sweep::Startup (SweepParams &sweepParams, SpinBlock& system, SpinBlock& newSystem)
{
  mcheck("at the start of block and decimate");
  dmrginp.guessgenT -> start();  // timer starts
  bool forward = (system.get_sites() [0] == 0); // if first site is 0, then it's forward sweep
  SpinBlock systemDot;
  // define the sites of "systemDot"
  int systemDotStart, systemDotEnd;
  int systemDotSize = sweepParams.get_sys_add() - 1;
  if (forward)
  {
    systemDotStart = *system.get_sites().rbegin () + 1;
    systemDotEnd = systemDotStart + systemDotSize;
  }
  else
  {
    systemDotStart = system.get_sites() [0] - 1;
    systemDotEnd = systemDotStart - systemDotSize;
  }

  vector<int> spindotsites(2); 
  spindotsites[0] = systemDotStart;
  spindotsites[1] = systemDotEnd;
  systemDot = SpinBlock(systemDotStart, systemDotEnd, true); // default is_complement=false
  
  const int nexact = forward ? sweepParams.get_forward_starting_size() : sweepParams.get_backward_starting_size();

  dmrginp.datatransfer -> start();
  system.addAdditionalCompOps(); // communicate between different processors, broadcast operators from system block
  dmrginp.datatransfer -> stop();
  InitBlocks::InitNewSystemBlock(system, systemDot, newSystem, sweepParams.current_root(), sweepParams.current_root(), sweepParams.get_sys_add(), dmrginp.direct(), 
				 DISTRIBUTED_STORAGE, true, true);

  int nquanta = newSystem.get_stateInfo().quanta.size();
  std::vector<DiagonalMatrix > energies(nquanta);
  std::vector<Matrix> rotateMatrix(nquanta);
  DensityMatrix transformmatrix; // FIXME pay attention to this: density matrix with certain quantum
  transformmatrix.allocate(newSystem.get_stateInfo());
  SpinQuantum q(0,SpinSpace(0),IrrepSpace(0));

  //if (mpigetrank() == 0) {
  double minval = 1e12;
  for (int i=0; i<nquanta; i++) {
    diagonalise(newSystem.get_op_rep(HAM,q)->operator_element(i,i), energies[i], transformmatrix(i,i));
    for (int j=0; j<energies[i].Nrows(); j++) 
      if (minval > energies[i](j+1))
	minval = energies[i](j+1);
  }
  if (mpigetrank() == 0) {
    for (int i=0; i<nquanta; i++) {
      for (int j=0; j<energies[i].Nrows(); j++) 
	energies[i](j+1) = 1.0/(energies[i](j+1)-minval+1);
    }


    vector<pair<int, int> > inorderwts;
    vector<vector<int> > wtsbyquanta;
    
    sort_weights(energies, inorderwts, wtsbyquanta);
    
    // make transformation matrix by various algorithms
    int keptstates = sweepParams.get_keep_states()/2, keptqstates = sweepParams.get_keep_states()-keptstates;
    int totalstatesbydm = min(static_cast<int>(inorderwts.size()), keptstates);
    int totalstatesbyquanta = min(static_cast<int>(inorderwts.size()), keptstates + keptqstates) - totalstatesbydm;
    if (totalstatesbyquanta < 0) totalstatesbyquanta = 0;
    
    pout << "\t\t\t total states using dm and quanta " << totalstatesbydm << " " << totalstatesbyquanta << endl;
    
    
    double error = assign_matrix_by_dm(rotateMatrix, energies, transformmatrix, inorderwts, wtsbyquanta, totalstatesbydm, totalstatesbyquanta, newSystem.size(), 2*totalstatesbydm);
    pout << "\t\t\t Total discarded weight "<<error<<endl;
  }

#ifndef SERIAL
  mpi::communicator world;
  broadcast(world, rotateMatrix, 0);
#endif

  dmrginp.operrotT -> start();
  newSystem.transform_operators(rotateMatrix);
  SaveRotationMatrix (newSystem.get_sites(), rotateMatrix);
  for (int i=0; i<dmrginp.nroots(); i++)
    SaveRotationMatrix (newSystem.get_sites(), rotateMatrix, i);
  dmrginp.operrotT -> stop();
  mcheck("after rotation and transformation of block");
  


  pout << dmrginp.guessgenT<<" "<<dmrginp.multiplierT<<" "<<dmrginp.operrotT<< "  "<<globaltimer.totalwalltime()<<" timer "<<endl;
  pout << dmrginp.makeopsT<<" makeops "<<endl;
  pout << dmrginp.datatransfer<<" datatransfer "<<endl;
  //cout << dmrginp.justmultiply<<" just multiply "<<endl;
  //cout << dmrginp.otherrotation<<" "<<dmrginp.spinrotation<<" "<<dmrginp.operrotT<<" rotations time "<<endl; 
  pout <<"oneindexopmult   twoindexopmult   Hc  couplingcoeff"<<endl;  
  pout << dmrginp.oneelecT<<" "<<dmrginp.twoelecT<<" "<<dmrginp.hmultiply<<" "<<dmrginp.couplingcoeff<<" hmult"<<endl;
  pout << dmrginp.buildsumblock<<" "<<dmrginp.buildblockops<<" build block"<<endl;
  pout << "addnoise  S_0_opxop  S_1_opxop   S_2_opxop"<<endl;
  pout << dmrginp.addnoise<<" "<<dmrginp.s0time<<" "<<dmrginp.s1time<<" "<<dmrginp.s2time<<endl;
  

  //mcheck("After renorm transform");
}


