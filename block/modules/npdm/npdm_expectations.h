/*                                                                           
Developed by Sandeep Sharma and Garnet K.-L. Chan, 2012                      
Copyright (c) 2012, Garnet K.-L. Chan                                        
                                                                             
This program is integrated in Molpro with the permission of 
Sandeep Sharma and Garnet K.-L. Chan
*/

#ifndef NPDM_EXPECT_H
#define NPDM_EXPECT_H

#include "npdm_patterns.h"
#include "npdm_expectations.h"
#include "npdm_spin_ops.h"
#include "npdm_spin_adaptation.h"

namespace SpinAdapted{
namespace Npdm{

//===========================================================================================================================================================

class Npdm_expectations {
 
  public:
    Npdm_expectations( Npdm_spin_adaptation& spin_adaptation, Npdm_patterns& npdm_patterns, 
                       const int order, Wavefunction & wavefunction, const SpinBlock & big );

    std::vector< std::pair< std::vector<int>, double > > 
      get_nonspin_adapted_expectations(NpdmSpinOps_base& lhsOps, NpdmSpinOps_base& rhsOps, NpdmSpinOps_base& dotOps );

  private:
    Npdm_spin_adaptation& spin_adaptation_;
    std::vector< double > expectations_;
    Npdm_patterns& npdm_patterns_;
    Wavefunction& wavefunction_; 
    const SpinBlock& big_; 
    const int npdm_order_;

    bool screen_op_string_for_duplicates( const std::string& op, const std::vector<int>& indices );
    double contract_spin_adapted_operators( int ilhs, int idot, int irhs, NpdmSpinOps_base& lhsOps, NpdmSpinOps_base& rhsOps, NpdmSpinOps_base& dotOps );
    void build_spin_adapted_singlet_expectations( NpdmSpinOps_base & lhsOps, NpdmSpinOps_base & rhsOps, NpdmSpinOps_base & dotOps );
    bool test_for_singlet( int ilhs, int idot, int irhs, NpdmSpinOps_base& lhsOps, NpdmSpinOps_base& rhsOps, NpdmSpinOps_base& dotOps );

    void get_full_op_string( NpdmSpinOps_base& lhsOps, NpdmSpinOps_base& rhsOps, NpdmSpinOps_base& dotOps, 
                             std::string& op_string, std::vector<int>& indices );

};

//===========================================================================================================================================================

}
}

#endif
