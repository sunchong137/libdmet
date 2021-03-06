/*                                                                           
Developed by Sandeep Sharma and Garnet K.-L. Chan, 2012                      
Copyright (c) 2012, Garnet K.-L. Chan                                        
                                                                             
This program is integrated in Molpro with the permission of 
Sandeep Sharma and Garnet K.-L. Chan
*/

#include "guess_wavefunction.h"
#include "sweepCompress.h"
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
#include "davidson.h"


using namespace boost;
using namespace std;


void SpinAdapted::SweepCompress::BlockDecimateAndCompress (SweepParams &sweepParams, SpinBlock& system, SpinBlock& newSystem, const bool &useSlater, const bool& dot_with_sys, int targetState, int baseState)
{
  int sweepiter = sweepParams.get_sweep_iter();
  if (dmrginp.outputlevel() > 0) {
    mcheck("at the start of block and decimate");
    pout << "\t\t\t dot with system "<<dot_with_sys<<endl;
  }
  pout <<endl<< "\t\t\t Performing Blocking"<<endl;
  // figure out if we are going forward or backwards
  dmrginp.guessgenT -> start();
  bool forward = (system.get_sites() [0] == 0);
  SpinBlock systemDot;
  SpinBlock envDot;
  int systemDotStart, systemDotEnd;
  int systemDotSize = sweepParams.get_sys_add() - 1;
  if (forward)
  {
    systemDotStart = dmrginp.spinAdapted() ? *system.get_sites().rbegin () + 1 : (*system.get_sites().rbegin ())/2 + 1 ;
    systemDotEnd = systemDotStart + systemDotSize;
  }
  else
  {
    systemDotStart = dmrginp.spinAdapted() ? system.get_sites()[0] - 1 : (system.get_sites()[0])/2 - 1 ;
    systemDotEnd = systemDotStart - systemDotSize;
  }
  vector<int> spindotsites(2); 
  spindotsites[0] = systemDotStart;
  spindotsites[1] = systemDotEnd;
  systemDot = SpinBlock(systemDotStart, systemDotEnd, false);
  SpinBlock environment, environmentDot, newEnvironment;


  int environmentDotStart, environmentDotEnd, environmentStart, environmentEnd;
  int environmentDotSize = sweepParams.get_env_add() -1;
  if (environmentDotSize <0) environmentDotSize = 0 ; 
  if (forward)
  {
    environmentDotStart = systemDotEnd + 1;
    environmentDotEnd = environmentDotStart + environmentDotSize;
  }
  else
  {
    environmentDotStart = systemDotEnd - 1;
    environmentDotEnd = environmentDotStart - environmentDotSize;
  }
  vector<int> envdotsites(2); 
  envdotsites[0] = environmentDotStart;
  envdotsites[1] = environmentDotEnd;

  if (!sweepParams.get_onedot())
    environmentDot = SpinBlock(environmentDotStart, environmentDotEnd, false);

  const int nexact = forward ? sweepParams.get_forward_starting_size() : sweepParams.get_backward_starting_size();

  //before halfway put the sysdot with system otherwise with environment
  if (!sweepParams.get_onedot()) {
      dmrginp.datatransfer -> start();
      system.addAdditionalCompOps();
      dmrginp.datatransfer -> stop();

      bool haveNormOps = dot_with_sys, haveCompOps = true;
      InitBlocks::InitNewSystemBlock(system, systemDot, newSystem, baseState, targetState, sweepParams.get_sys_add(), dmrginp.direct(), 
      			     DISTRIBUTED_STORAGE, haveNormOps, haveCompOps);
      if (dmrginp.outputlevel() > 0)
         mcheck("");

      InitBlocks::InitNewEnvironmentBlock(environment, environmentDot, newEnvironment, system, systemDot, baseState, targetState,
					  sweepParams.get_sys_add(), sweepParams.get_env_add(), forward, dmrginp.direct(),
					  sweepParams.get_onedot(), nexact, useSlater, !haveNormOps, haveCompOps, dot_with_sys);
      if (dmrginp.outputlevel() > 0)
         mcheck("");
  }
  else {
    dmrginp.datatransfer -> start();
    system.addAdditionalCompOps();
    dmrginp.datatransfer -> stop();

    bool haveNormOps = dot_with_sys, haveCompOps = true;
    if (dot_with_sys) {
      InitBlocks::InitNewSystemBlock(system, systemDot, newSystem, baseState, targetState, sweepParams.get_sys_add(), dmrginp.direct(), DISTRIBUTED_STORAGE, haveNormOps, haveCompOps);

    }
    InitBlocks::InitNewEnvironmentBlock(environment, systemDot, newEnvironment, system, systemDot, baseState, targetState,
					sweepParams.get_sys_add(), sweepParams.get_env_add(), forward, dmrginp.direct(),
					sweepParams.get_onedot(), nexact, useSlater, !haveNormOps, haveCompOps, dot_with_sys);
  }
  SpinBlock big;  // new_sys = sys+sys_dot; new_env = env+env_dot; big = new_sys + new_env then renormalize to find new_sys(new)
  if (dot_with_sys) {
    newSystem.set_loopblock(true);
    system.set_loopblock(false);
    newEnvironment.set_loopblock(false);
    if (!sweepParams.get_onedot())
      environment.set_loopblock(false);
    InitBlocks::InitBigBlock(newSystem, newEnvironment, big); 
  }
  else{
    if (sweepParams.get_onedot()) {
      system.set_loopblock(false);
      newEnvironment.set_loopblock(true);
      environment.set_loopblock(true);
      InitBlocks::InitBigBlock(system, newEnvironment, big); 
    }
    else {
      newSystem.set_loopblock(false);
      system.set_loopblock(false);
      newEnvironment.set_loopblock(true);
      environment.set_loopblock(false);
      InitBlocks::InitBigBlock(newSystem, newEnvironment, big); 
    }
  }
  //analyse_operator_distribution(big);
  dmrginp.guessgenT -> stop();
  dmrginp.multiplierT -> start();
  std::vector<Matrix> rotatematrix;

  if (dmrginp.outputlevel() > 0)
    mcheck(""); 
  if (dmrginp.outputlevel() == 0) {
    if (!dot_with_sys && sweepParams.get_onedot()) {
      pout << "\t\t\t System  Block"<<system;
      pout << "\t\t\t Environment Block"<<newEnvironment<<endl;
    }
    else {
      pout << "\t\t\t System  Block"<<newSystem;
      pout << "\t\t\t Environment Block"<<newEnvironment<<endl;
    }
    pout << "\t\t\t Solving wavefunction "<<endl;
  }

  std::vector<Wavefunction> solution; solution.resize(1);

  DiagonalMatrix e;
  e.ReSize(big.get_stateInfo().totalStates); e= 0;

  //read the 0th wavefunction which we keep on the ket side because by default the ket stateinfo is used to initialize wavefunction
  //also when you use spinblock operators to multiply a state, it does so from the ket side i.e.  H|ket>
  GuessWave::guess_wavefunctions(solution, e, big, sweepParams.set_guesstype(), sweepParams.get_onedot(), dot_with_sys, 0.0, baseState); 

  multiply_h davidson_f(big, sweepParams.get_onedot());
  vector<Wavefunction> outputState; outputState.resize(1);
  outputState[0].AllowQuantaFor(big.get_leftBlock()->get_braStateInfo(), big.get_rightBlock()->get_braStateInfo(), dmrginp.effective_molecule_quantum_vec());
  outputState[0].set_onedot(sweepParams.get_onedot());
  outputState[0].Clear();

  davidson_f(solution[0], outputState[0]);

  Wavefunction overlapIntermediate=outputState[0]; overlapIntermediate.Clear();
  big.multiplyOverlap(solution[0], &overlapIntermediate, MAX_THRD);
  double overlap = DotProduct(overlapIntermediate, outputState[0])/sqrt(DotProduct(outputState[0], outputState[0])* DotProduct(solution[0], solution[0]));
  pout << "\t\t\t The overlap = "<<overlap<<endl;
  
  sweepParams.set_lowest_energy() = std::vector<double>(1,overlap);

  SpinBlock newbig;

  if (sweepParams.get_onedot() && !dot_with_sys)
  {
    InitBlocks::InitNewSystemBlock(system, systemDot, newSystem, baseState, targetState, systemDot.size(), dmrginp.direct(), DISTRIBUTED_STORAGE, false, true);
    InitBlocks::InitBigBlock(newSystem, environment, newbig); 

    Wavefunction tempwave = solution[0];
    GuessWave::onedot_shufflesysdot(big.get_ketStateInfo(), newbig.get_ketStateInfo(), solution[0], tempwave);  
    solution[0] = tempwave;

    tempwave = outputState[0];
    GuessWave::onedot_shufflesysdot(big.get_braStateInfo(), newbig.get_braStateInfo(), outputState[0], tempwave);  
    outputState[0] = tempwave;

    big.get_rightBlock()->clear();
    big.clear();
  }
  else
    newbig = big;

  environment.clear();
  newEnvironment.clear();


  std::vector<Matrix> brarotateMatrix, ketrotateMatrix;
  DensityMatrix bratracedMatrix(newSystem.get_braStateInfo()), kettracedMatrix(newSystem.get_ketStateInfo());
  bratracedMatrix.allocate(newSystem.get_braStateInfo()); kettracedMatrix.allocate(newSystem.get_ketStateInfo());

  bratracedMatrix.makedensitymatrix(outputState, newbig, dmrginp.weights(sweepiter), 0.0, 0.0, true);
  kettracedMatrix.makedensitymatrix(solution, newbig, dmrginp.weights(sweepiter), 0.0, 0.0, true);
  double braerror, keterror;
  if (!mpigetrank()) {
    keterror = makeRotateMatrix(kettracedMatrix, ketrotateMatrix, newbig.get_rightBlock()->get_ketStateInfo().totalStates, 0);
    braerror = makeRotateMatrix(bratracedMatrix, brarotateMatrix, sweepParams.get_keep_states(), sweepParams.get_keep_qstates());
  }

#ifndef SERIAL
  mpi::communicator world;
  broadcast(world, ketrotateMatrix, 0);
  broadcast(world, brarotateMatrix, 0);
#endif

  //assert(keterror < NUMERICAL_ZERO);
  pout << "\t\t\t Total ket discarded weight "<<keterror<<endl<<endl;
  pout << "\t\t\t Total bra discarded weight "<<braerror<<endl<<endl;

  sweepParams.set_lowest_error() = braerror;

  SaveRotationMatrix (newbig.get_leftBlock()->get_sites(), ketrotateMatrix, baseState);
  SaveRotationMatrix (newbig.get_leftBlock()->get_sites(), brarotateMatrix, targetState);
  solution[0].SaveWavefunctionInfo (newbig.get_ketStateInfo(), newbig.get_leftBlock()->get_sites(), baseState);
  outputState[0].SaveWavefunctionInfo (newbig.get_braStateInfo(), newbig.get_leftBlock()->get_sites(), targetState);

  pout <<"\t\t\t Performing Renormalization "<<endl;
  newSystem.transform_operators(brarotateMatrix, ketrotateMatrix);




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

double SpinAdapted::SweepCompress::do_one(SweepParams &sweepParams, const bool &warmUp, const bool &forward, const bool &restart, const int &restartSize, int targetState, int baseState)
{

  SpinBlock system;
  const int nroots = dmrginp.nroots(sweepParams.get_sweep_iter());

  std::vector<double> finalEnergy(nroots,-1.0e10);
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

  InitBlocks::InitStartingBlock (system,forward, baseState, targetState, sweepParams.get_forward_starting_size(), sweepParams.get_backward_starting_size(), restartSize, restart, warmUp);
  if(!restart)
    sweepParams.set_block_iter() = 0;

 
  if (dmrginp.outputlevel() > 0)
    pout << "\t\t\t Starting block is :: " << endl << system << endl;

  SpinBlock::store (forward, system.get_sites(), system, baseState, targetState); // if restart, just restoring an existing block --
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

      if (sweepParams.get_block_iter() != 0) 
	sweepParams.set_guesstype() = TRANSFORM;
      else
        sweepParams.set_guesstype() = TRANSPOSE;


      
      if (dmrginp.outputlevel() > 0)
         pout << "\t\t\t Blocking and Decimating " << endl;
	  
      SpinBlock newSystem; // new system after blocking and decimating

      //Need to substitute by:
      if (warmUp )
	Startup(sweepParams, system, newSystem, dot_with_sys, targetState, baseState);
      else {
	BlockDecimateAndCompress (sweepParams, system, newSystem, warmUp, dot_with_sys, targetState, baseState);
      }
      
      //Need to substitute by?

      if (!warmUp ){

	//this criteria should work for state average or state specific because the lowest sweep energy is always the lowest of the average
	finalError = max(sweepParams.get_lowest_error(),finalError);
	finalEnergy[0] = max(sweepParams.get_lowest_energy()[0], finalEnergy[0]);
	pout << "final energy "<<finalEnergy[0]<<"  "<<sweepParams.get_lowest_energy()[0]<<endl;
      }
      
      system = newSystem;
      if (dmrginp.outputlevel() > 0){
	    pout << system<<endl;
	    system.printOperatorSummary();
      }
      
      //system size is going to be less than environment size
      if (forward && system.get_complementary_sites()[0] >= dmrginp.last_site()/2)
	    dot_with_sys = false;
      if (!forward && system.get_sites()[0]-1 < dmrginp.last_site()/2)
	    dot_with_sys = false;

      SpinBlock::store (forward, system.get_sites(), system, baseState, targetState);	 	
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


  pout << "\t\t\t Largest Error for Sweep with " << sweepParams.get_keep_states() << " states is " << finalError << endl;
  pout << "\t\t\t Largest overlap for Sweep with " << sweepParams.get_keep_states() << " states is " << finalEnergy[0] << endl;
  sweepParams.set_largest_dw() = finalError;
  

  pout << "\t\t\t ============================================================================ " << endl;

  // update the static number of iterations

  ++sweepParams.set_sweep_iter();

  return finalError;
}


void SpinAdapted::SweepCompress::Startup (SweepParams &sweepParams, SpinBlock& system, SpinBlock& newSystem, const bool& dot_with_sys, int targetState, int baseState)
{
  bool useSlater = false;
  pout <<endl<< "\t\t\t Performing Blocking"<<endl;
  // figure out if we are going forward or backwards
  dmrginp.guessgenT -> start();
  bool forward = (system.get_sites() [0] == 0);
  SpinBlock systemDot;
  SpinBlock envDot;
  int systemDotStart, systemDotEnd;
  int systemDotSize = sweepParams.get_sys_add() - 1;
  if (forward)
  {
    systemDotStart = dmrginp.spinAdapted() ? *system.get_sites().rbegin () + 1 : (*system.get_sites().rbegin ())/2 + 1 ;
    systemDotEnd = systemDotStart + systemDotSize;
  }
  else
  {
    systemDotStart = dmrginp.spinAdapted() ? system.get_sites()[0] - 1 : (system.get_sites()[0])/2 - 1 ;
    systemDotEnd = systemDotStart - systemDotSize;
  }
  vector<int> spindotsites(2); 
  spindotsites[0] = systemDotStart;
  spindotsites[1] = systemDotEnd;
  systemDot = SpinBlock(systemDotStart, systemDotEnd, false);
  SpinBlock environment, environmentDot, newEnvironment;


  int environmentDotStart, environmentDotEnd, environmentStart, environmentEnd;
  int environmentDotSize = sweepParams.get_env_add() -1;
  if (environmentDotSize <0) environmentDotSize = 0 ; 
  if (forward)
  {
    environmentDotStart = systemDotEnd + 1;
    environmentDotEnd = environmentDotStart + environmentDotSize;
  }
  else
  {
    environmentDotStart = systemDotEnd - 1;
    environmentDotEnd = environmentDotStart - environmentDotSize;
  }
  vector<int> envdotsites(2); 
  envdotsites[0] = environmentDotStart;
  envdotsites[1] = environmentDotEnd;

  if (!sweepParams.get_onedot())
    environmentDot = SpinBlock(environmentDotStart, environmentDotEnd, false);

  const int nexact = forward ? sweepParams.get_forward_starting_size() : sweepParams.get_backward_starting_size();

  //before halfway put the sysdot with system otherwise with environment
  if (!sweepParams.get_onedot()) {
      dmrginp.datatransfer -> start();
      system.addAdditionalCompOps();
      dmrginp.datatransfer -> stop();

      bool haveNormOps = dot_with_sys, haveCompOps = true;
      InitBlocks::InitNewSystemBlock(system, systemDot, newSystem, baseState, targetState, sweepParams.get_sys_add(), dmrginp.direct(), 
      			     DISTRIBUTED_STORAGE, haveNormOps, haveCompOps);
      if (dmrginp.outputlevel() > 0)
         mcheck("");

      InitBlocks::InitNewEnvironmentBlock(environment, environmentDot, newEnvironment, system, systemDot, -1, -1,
					  sweepParams.get_sys_add(), sweepParams.get_env_add(), forward, dmrginp.direct(),
					  sweepParams.get_onedot(), nexact, useSlater, !haveNormOps, haveCompOps, dot_with_sys);
      if (dmrginp.outputlevel() > 0)
         mcheck("");
  }
  else {
    dmrginp.datatransfer -> start();
    system.addAdditionalCompOps();
    dmrginp.datatransfer -> stop();

    bool haveNormOps = dot_with_sys, haveCompOps = true;
    if (dot_with_sys) {
      InitBlocks::InitNewSystemBlock(system, systemDot, newSystem, targetState, baseState, sweepParams.get_sys_add(), dmrginp.direct(), DISTRIBUTED_STORAGE, haveNormOps, haveCompOps);

    }
    InitBlocks::InitNewEnvironmentBlock(environment, systemDot, newEnvironment, system, systemDot, -1, -1,
					sweepParams.get_sys_add(), sweepParams.get_env_add(), forward, dmrginp.direct(),
					sweepParams.get_onedot(), nexact, useSlater, !haveNormOps, haveCompOps, dot_with_sys);
  }
  SpinBlock big;  // new_sys = sys+sys_dot; new_env = env+env_dot; big = new_sys + new_env then renormalize to find new_sys(new)
  if (dot_with_sys) {
    newSystem.set_loopblock(true);
    system.set_loopblock(false);
    newEnvironment.set_loopblock(false);
    if (!sweepParams.get_onedot())
      environment.set_loopblock(false);
    InitBlocks::InitBigBlock(newSystem, newEnvironment, big); 
  }
  else{
    if (sweepParams.get_onedot()) {
      system.set_loopblock(false);
      newEnvironment.set_loopblock(true);
      environment.set_loopblock(true);
      InitBlocks::InitBigBlock(system, newEnvironment, big); 
    }
    else {
      newSystem.set_loopblock(false);
      system.set_loopblock(false);
      newEnvironment.set_loopblock(true);
      environment.set_loopblock(false);
      InitBlocks::InitBigBlock(newSystem, newEnvironment, big); 
    }
  }
  //analyse_operator_distribution(big);
  dmrginp.guessgenT -> stop();
  dmrginp.multiplierT -> start();
  std::vector<Matrix> rotatematrix;

  if (dmrginp.outputlevel() > 0)
    mcheck(""); 
  if (dmrginp.outputlevel() == 0) {
    if (!dot_with_sys && sweepParams.get_onedot()) {
      pout << "\t\t\t System  Block"<<system;
      pout << "\t\t\t Environment Block"<<newEnvironment<<endl;
    }
    else {
      pout << "\t\t\t System  Block"<<newSystem;
      pout << "\t\t\t Environment Block"<<newEnvironment<<endl;
    }
    pout << "\t\t\t Solving wavefunction "<<endl;
  }

  std::vector<Wavefunction> solution; solution.resize(1);

  DiagonalMatrix e;
  e.ReSize(big.get_stateInfo().totalStates); e= 0;
  int guessState = 0;

  //read the 0th wavefunction which we keep on the ket side because by default the ket stateinfo is used to initialize wavefunction
  //also when you use spinblock operators to multiply a state, it does so from the ket side i.e.  H|ket>
  GuessWave::guess_wavefunctions(solution, e, big, sweepParams.set_guesstype(), sweepParams.get_onedot(), dot_with_sys, baseState, targetState); 

  SpinBlock newbig;

  if (sweepParams.get_onedot() && !dot_with_sys)
  {
    InitBlocks::InitNewSystemBlock(system, systemDot, newSystem, baseState, guessState, systemDot.size(), dmrginp.direct(), DISTRIBUTED_STORAGE, false, true);
    InitBlocks::InitBigBlock(newSystem, environment, newbig); 

    Wavefunction tempwave = solution[0];
    tempwave.Clear();
    GuessWave::onedot_shufflesysdot(big.get_ketStateInfo(), newbig.get_ketStateInfo(), solution[0], tempwave);  
    solution[0] = tempwave;

    big.get_rightBlock()->clear();
    big.clear();
  }
  else
    newbig = big;


  environment.clear();
  newEnvironment.clear();

  std::vector<Matrix> ketrotateMatrix;
  DensityMatrix kettracedMatrix(newSystem.get_ketStateInfo());
  kettracedMatrix.allocate(newSystem.get_ketStateInfo());

  kettracedMatrix.makedensitymatrix(solution, newbig, dmrginp.weights(0), 0.0, 0.0, true);
  double keterror;
  if (!mpigetrank()) {
    keterror = makeRotateMatrix(kettracedMatrix, ketrotateMatrix, newbig.get_rightBlock()->get_ketStateInfo().totalStates, 0);
  }

#ifndef SERIAL
  mpi::communicator world;
  broadcast(world, ketrotateMatrix, 0);
#endif

  //assert(keterror < NUMERICAL_ZERO);
  pout <<"\t\t\t Performing Renormalization "<<endl;
  pout << "\t\t\t Total discarded weight "<<keterror<<endl<<endl;
  sweepParams.set_lowest_error() = keterror;

  SaveRotationMatrix (newbig.get_leftBlock()->get_sites(), ketrotateMatrix, baseState);
  SaveRotationMatrix (newbig.get_leftBlock()->get_sites(), ketrotateMatrix, targetState);
  solution[0].SaveWavefunctionInfo (newbig.get_ketStateInfo(), newbig.get_leftBlock()->get_sites(), baseState);
  solution[0].SaveWavefunctionInfo (newbig.get_ketStateInfo(), newbig.get_leftBlock()->get_sites(), targetState);


  newSystem.transform_operators(ketrotateMatrix);




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
