/*                                                                           
Developed by Sandeep Sharma and Garnet K.-L. Chan, 2012                      
Copyright (c) 2012, Garnet K.-L. Chan                                        
                                                                             
This program is integrated in Molpro with the permission of 
Sandeep Sharma and Garnet K.-L. Chan
*/

#ifndef NPDM_EXPECTATIONS_ENGINE_H
#define NPDM_EXPECTATIONS_ENGINE_H

#include "spinblock.h"
#include "wavefunction.h"
#include "BaseOperator.h"

namespace SpinAdapted{
namespace Npdm{

  void FormLeftOp(const SpinBlock* leftBlock, const boost::shared_ptr<SparseMatrix> leftOp, const boost::shared_ptr<SparseMatrix> dotOp, SparseMatrix& Aop, int totalspin);
  double DotProduct(const Wavefunction& w1, const Wavefunction& w2, double Sz, const SpinBlock& big);
  double spinExpectation(Wavefunction& wave1, Wavefunction& wave2, boost::shared_ptr<SparseMatrix> leftOp, boost::shared_ptr<SparseMatrix> dotOp, boost::shared_ptr<SparseMatrix> rightOp, const SpinBlock& big);

}
}

#endif

