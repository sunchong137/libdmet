/*                                                                           
Developed by Sandeep Sharma and Garnet K.-L. Chan, 2012                      
Copyright (c) 2012, Garnet K.-L. Chan                                        
                                                                             
This program is integrated in Molpro with the permission of 
Sandeep Sharma and Garnet K.-L. Chan
*/

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//
// Extension of Operators.C for 3-index operators
//FIXME there's a lot of duplication, especially in build_from_disk... Templates??
//
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

#include "op_components.h"
#include "BaseOperator.h"
#include "spinblock.h"
#include "operatorfunctions.h"
#include "tensor_operator.h"
#include "three_index_ops.h"

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//  (Des,Des,Des)
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

double SpinAdapted::DesDesDes::redMatrixElement(Csf c1, vector<Csf>& ladder, const SpinBlock* b)
{
  assert( build_pattern == "((DD)(D))" );
  double element = 0.0;
  int I = get_orbs()[0]; 
  int J = get_orbs()[1];
  int K = get_orbs()[2];

  // Must take into account how the 3-index is built from a combination of 2-index and 1-index
  std::vector<SpinQuantum> quantum_ladder = get_quantum_ladder().at("((DD)(D))");
  assert( quantum_ladder.size() == 2 );
  SpinQuantum deltaQuantum12 = quantum_ladder.at(0);
  SpinQuantum deltaQuantum123 = quantum_ladder.at(1);
  deltaQuantum[0] = deltaQuantum123;

  // Spin quantum data for first pair of operators combined
  IrrepSpace sym12 = deltaQuantum12.get_symm();
  int irrep12 = deltaQuantum12.get_symm().getirrep();
  int spin12 = deltaQuantum12.get_s().getirrep();
  // Spin quantum data for total operator
  IrrepSpace sym123 = deltaQuantum123.get_symm();
  int irrep123 = deltaQuantum123.get_symm().getirrep();
  int spin123 = deltaQuantum123.get_s().getirrep();

  TensorOp D1(I,-1); 
  TensorOp D2(J,-1); 
  TensorOp D3(K,-1); 

  // Combine first two operators
  TensorOp DD = D1.product(D2, spin12, irrep12);
  // Combine with third operator
  TensorOp DDD = DD.product(D3, spin123, irrep123);

  for (int i=0; i<ladder.size(); i++)
  {
    int index = 0; double cleb=0.0;
    if (nonZeroTensorComponent(c1, deltaQuantum[0], ladder[i], index, cleb)) {
      std::vector<double> MatElements = calcMatrixElements(c1, DDD, ladder[i]) ;
      element = MatElements[index]/cleb;
      break;
    }
    else
      continue;
  }
  return element;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

boost::shared_ptr<SpinAdapted::SparseMatrix> SpinAdapted::DesDesDes::getworkingrepresentation(const SpinBlock* block)
{
  assert(this->get_initialised());
  if (this->get_built()) {
    return boost::shared_ptr<DesDesDes>(this, boostutils::null_deleter()); // boost::shared_ptr does not own op
  }
  else {
    boost::shared_ptr<SparseMatrix> rep(new DesDesDes);
    *rep = *this;
    rep->build(*block);

    return rep;
  }
}

//===========================================================================================================================================================
// 3PDM operators
//===========================================================================================================================================================

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//  (Cre,Cre,Des)
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

double SpinAdapted::CreCreDes::redMatrixElement(Csf c1, vector<Csf>& ladder, const SpinBlock* b)
{
//cout << "building CreCreDes explicitly from CSF..\n";
  assert( build_pattern == "((CC)(D))" );
  double element = 0.0;
  int I = get_orbs()[0]; 
  int J = get_orbs()[1];
  int K = get_orbs()[2];

  // Must take into account how the 3-index is built from a combination of 2-index and 1-index
  std::vector<SpinQuantum> quantum_ladder = get_quantum_ladder().at("((CC)(D))");
  assert( quantum_ladder.size() == 2 );
  SpinQuantum deltaQuantum12 = quantum_ladder.at(0);
  SpinQuantum deltaQuantum123 = quantum_ladder.at(1);
  deltaQuantum[0] = deltaQuantum123;

  // Spin quantum data for first pair of operators combined
  IrrepSpace sym12 = deltaQuantum12.get_symm();
  int irrep12 = deltaQuantum12.get_symm().getirrep();
  int spin12 = deltaQuantum12.get_s().getirrep();
  // Spin quantum data for total operator
  IrrepSpace sym123 = deltaQuantum123.get_symm();
  int irrep123 = deltaQuantum123.get_symm().getirrep();
  int spin123 = deltaQuantum123.get_s().getirrep();

//pout << "redMatrixElement indices:\n";
//pout << I << "  " << J << "  " << K << std::endl;
//pout << "spin composition:\n";
//pout << spin12/2.0 << "  " << spin123/2.0 << std::endl;
  TensorOp C1(I, 1); 
  TensorOp C2(J, 1); 
  TensorOp D3(K,-1); 

  // Combine first two operators
//FIXME  TensorOp CC = C1.product(C2, spin12, irrep12);  I==J has no affect
  TensorOp CC = C1.product(C2, spin12, irrep12, I==J);
  // Combine with third operator
  TensorOp CCD = CC.product(D3, spin123, irrep123);

  for (int i=0; i<ladder.size(); i++)
  {
    int index = 0; double cleb=0.0;
    if (nonZeroTensorComponent(c1, deltaQuantum[0], ladder[i], index, cleb)) {
      std::vector<double> MatElements = calcMatrixElements(c1, CCD, ladder[i]) ;
      element = MatElements[index]/cleb;
      break;
    }
    else
      continue;
  }
  return element;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

boost::shared_ptr<SpinAdapted::SparseMatrix> SpinAdapted::CreCreDes::getworkingrepresentation(const SpinBlock* block)
{
  assert(this->get_initialised());
  if (this->get_built()) {
    return boost::shared_ptr<CreCreDes>(this, boostutils::null_deleter()); // boost::shared_ptr does not own op
  }
  else {
    boost::shared_ptr<SparseMatrix> rep(new CreCreDes);
    *rep = *this;
    rep->build(*block);
    return rep;
  }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//  (Cre,Des,Des)
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

double SpinAdapted::CreDesDes::redMatrixElement(Csf c1, vector<Csf>& ladder, const SpinBlock* b)
{
//cout << "building CreDesDes explicitly from CSF..\n";
  assert( build_pattern == "((CD)(D))" );
  double element = 0.0;
  int I = get_orbs()[0]; 
  int J = get_orbs()[1];
  int K = get_orbs()[2];

  // Must take into account how the 3-index is built from a combination of 2-index and 1-index
  std::vector<SpinQuantum> quantum_ladder = get_quantum_ladder().at("((CD)(D))");
  assert( quantum_ladder.size() == 2 );
  SpinQuantum deltaQuantum12 = quantum_ladder.at(0);
  SpinQuantum deltaQuantum123 = quantum_ladder.at(1);
  deltaQuantum[0] = deltaQuantum123;

  // Spin quantum data for first pair of operators combined
  IrrepSpace sym12 = deltaQuantum12.get_symm();
  int irrep12 = deltaQuantum12.get_symm().getirrep();
  int spin12 = deltaQuantum12.get_s().getirrep();
  // Spin quantum data for total operator
  IrrepSpace sym123 = deltaQuantum123.get_symm();
  int irrep123 = deltaQuantum123.get_symm().getirrep();
  int spin123 = deltaQuantum123.get_s().getirrep();

//FIXME
//pout << "redMatrixElement indices:\n";
//pout << I << "  " << J << "  " << K << std::endl;
//pout << "spin composition:\n";
//pout << spin12/2.0 << "  " << spin123/2.0 << std::endl;
  TensorOp C1(I, 1); 
  TensorOp D2(J,-1); 
  TensorOp D3(K,-1); 

  // Combine first two operators
  TensorOp CD = C1.product(D2, spin12, irrep12);
  // Combine with third operator
  TensorOp CDD = CD.product(D3, spin123, irrep123);

  for (int i=0; i<ladder.size(); i++)
  {
    int index = 0; double cleb=0.0;
    if (nonZeroTensorComponent(c1, deltaQuantum[0], ladder[i], index, cleb)) {
      std::vector<double> MatElements = calcMatrixElements(c1, CDD, ladder[i]) ;
      element = MatElements[index]/cleb;
      break;
    }
    else
      continue;
  }
  return element;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

boost::shared_ptr<SpinAdapted::SparseMatrix> SpinAdapted::CreDesDes::getworkingrepresentation(const SpinBlock* block)
{
  assert(this->get_initialised());
  if (this->get_built()) {
    return boost::shared_ptr<CreDesDes>(this, boostutils::null_deleter()); // boost::shared_ptr does not own op
  }
  else {
    boost::shared_ptr<SparseMatrix> rep(new CreDesDes);
    *rep = *this;
    rep->build(*block);

    return rep;
  }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//  (Cre,Des,Cre)
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

double SpinAdapted::CreDesCre::redMatrixElement(Csf c1, vector<Csf>& ladder, const SpinBlock* b)
{
//cout << "building CreDesCre in CSF space explicitly..\n";
  assert( build_pattern == "((CD)(C))" );
  double element = 0.0;
  int I = get_orbs()[0]; 
  int J = get_orbs()[1];
  int K = get_orbs()[2];

  // Must take into account how the 3-index is built from a combination of 2-index and 1-index
  std::vector<SpinQuantum> quantum_ladder = get_quantum_ladder().at("((CD)(C))");
  assert( quantum_ladder.size() == 2 );
  SpinQuantum deltaQuantum12 = quantum_ladder.at(0);
  SpinQuantum deltaQuantum123 = quantum_ladder.at(1);
  assert( deltaQuantum[0] == deltaQuantum123 );

  // Spin quantum data for first pair of operators combined
  IrrepSpace sym12 = deltaQuantum12.get_symm();
  int irrep12 = deltaQuantum12.get_symm().getirrep();
  int spin12 = deltaQuantum12.get_s().getirrep();
  // Spin quantum data for total operator
  IrrepSpace sym123 = deltaQuantum123.get_symm();
  int irrep123 = deltaQuantum123.get_symm().getirrep();
  int spin123 = deltaQuantum123.get_s().getirrep();

//pout << "redMatrixElement indices:\n";
//pout << I << "  " << J << "  " << K << std::endl;
//pout << "spin composition:\n";
//pout << spin12/2.0 << "  " << spin123/2.0 << std::endl;
  TensorOp C1(I, 1); 
  TensorOp D2(J,-1); 
  TensorOp C3(K, 1); 

  // Combine first two operators
  TensorOp CD = C1.product(D2, spin12, irrep12);
  // Combine with third operator
  TensorOp CDC = CD.product(C3, spin123, irrep123);

  for (int i=0; i<ladder.size(); i++)
  {
    int index = 0; double cleb=0.0;
    if (nonZeroTensorComponent(c1, deltaQuantum[0], ladder[i], index, cleb)) {
      std::vector<double> MatElements = calcMatrixElements(c1, CDC, ladder[i]) ;
      element = MatElements[index]/cleb;
      break;
    }
    else
      continue;
  }
  return element;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

boost::shared_ptr<SpinAdapted::SparseMatrix> SpinAdapted::CreDesCre::getworkingrepresentation(const SpinBlock* block)
{
  assert(this->get_initialised());
  if (this->get_built()) {
    return boost::shared_ptr<CreDesCre>(this, boostutils::null_deleter()); // boost::shared_ptr does not own op
  }
  else {
    boost::shared_ptr<SparseMatrix> rep(new CreDesCre);
    *rep = *this;
    rep->build(*block);
    return rep;
  }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//  (Cre,Cre,Cre)
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

double SpinAdapted::CreCreCre::redMatrixElement(Csf c1, vector<Csf>& ladder, const SpinBlock* b)
{
//cout << "building CreCreCre explicitly from CSF..\n";
//cout << "mpirank = " << mpigetrank() << endl;
  assert( build_pattern == "((CC)(C))" );
  double element = 0.0;
  int I = get_orbs()[0]; 
  int J = get_orbs()[1];
  int K = get_orbs()[2];
//cout << "i,j,k = " << I << " " << J << " " << K << endl;

  // Must take into account how the 3-index is built from a combination of 2-index and 1-index
  std::vector<SpinQuantum> quantum_ladder = get_quantum_ladder().at("((CC)(C))");
  assert( quantum_ladder.size() == 2 );
  SpinQuantum deltaQuantum12 = quantum_ladder.at(0);
  SpinQuantum deltaQuantum123 = quantum_ladder.at(1);
//FIXME components != 0
  deltaQuantum[0] = deltaQuantum123;

  // Spin quantum data for first pair of operators combined
  IrrepSpace sym12 = deltaQuantum12.get_symm();
  int irrep12 = deltaQuantum12.get_symm().getirrep();
  int spin12 = deltaQuantum12.get_s().getirrep();
  // Spin quantum data for total operator
  IrrepSpace sym123 = deltaQuantum123.get_symm();
  int irrep123 = deltaQuantum123.get_symm().getirrep();
  int spin123 = deltaQuantum123.get_s().getirrep();

  TensorOp C1(I,1); 
  TensorOp C2(J,1); 
  TensorOp C3(K,1); 

  // Combine first two operators
//FIXME I=J argument
  TensorOp CC = C1.product(C2, spin12, irrep12);
  // Combine with third operator
  TensorOp CCC = CC.product(C3, spin123, irrep123);

//FIXME loop over deltaQuantum components
  int j = 0;
  for (int i=0; i<ladder.size(); i++)
  {
    int index = 0; double cleb=0.0;
    if (nonZeroTensorComponent(c1, deltaQuantum[j], ladder[i], index, cleb)) {
      std::vector<double> MatElements = calcMatrixElements(c1, CCC, ladder[i]) ;
      element = MatElements[index]/cleb;
      break;
    }
    else
      continue;
  }
  return element;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

boost::shared_ptr<SpinAdapted::SparseMatrix> SpinAdapted::CreCreCre::getworkingrepresentation(const SpinBlock* block)
{
//pout << "CreCreCre::getworkingrepresentation\n";
  assert(this->get_initialised());
  if (this->get_built()) {
//pout << "get CCC from memory\n";
    return boost::shared_ptr<CreCreCre>(this, boostutils::null_deleter()); // boost::shared_ptr does not own op
  }
  else {
//pout << "build CCC\n";
    boost::shared_ptr<SparseMatrix> rep(new CreCreCre);
    *rep = *this;
    rep->build(*block);

    return rep;
  }
}

//===========================================================================================================================================================
// 4PDM operators
//===========================================================================================================================================================

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//  (Des,Cre,Des)
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

double SpinAdapted::DesCreDes::redMatrixElement(Csf c1, vector<Csf>& ladder, const SpinBlock* b)
{
  assert( build_pattern == "((DC)(D))" );
  double element = 0.0;
  int I = get_orbs()[0]; 
  int J = get_orbs()[1];
  int K = get_orbs()[2];

  // Must take into account how the 3-index is built from a combination of 2-index and 1-index
  std::vector<SpinQuantum> quantum_ladder = get_quantum_ladder().at("((DC)(D))");
  assert( quantum_ladder.size() == 2 );
  SpinQuantum deltaQuantum12 = quantum_ladder.at(0);
  SpinQuantum deltaQuantum123 = quantum_ladder.at(1);
  deltaQuantum[0] = deltaQuantum123;

  // Spin quantum data for first pair of operators combined
  IrrepSpace sym12 = deltaQuantum12.get_symm();
  int irrep12 = deltaQuantum12.get_symm().getirrep();
  int spin12 = deltaQuantum12.get_s().getirrep();
  // Spin quantum data for total operator
  IrrepSpace sym123 = deltaQuantum123.get_symm();
  int irrep123 = deltaQuantum123.get_symm().getirrep();
  int spin123 = deltaQuantum123.get_s().getirrep();

  TensorOp D1(I,-1); 
  TensorOp C2(J, 1); 
  TensorOp D3(K,-1); 

  // Combine first two operators
  TensorOp DC = D1.product(C2, spin12, irrep12);
  // Combine with third operator
  TensorOp DCD = DC.product(D3, spin123, irrep123);

  for (int i=0; i<ladder.size(); i++)
  {
    int index = 0; double cleb=0.0;
    if (nonZeroTensorComponent(c1, deltaQuantum[0], ladder[i], index, cleb)) {
      std::vector<double> MatElements = calcMatrixElements(c1, DCD, ladder[i]) ;
      element = MatElements[index]/cleb;
      break;
    }
    else
      continue;
  }
  return element;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

boost::shared_ptr<SpinAdapted::SparseMatrix> SpinAdapted::DesCreDes::getworkingrepresentation(const SpinBlock* block)
{
  assert(this->get_initialised());
  if (this->get_built()) {
    return boost::shared_ptr<DesCreDes>(this, boostutils::null_deleter()); // boost::shared_ptr does not own op
  }
  else {
    boost::shared_ptr<SparseMatrix> rep(new DesCreDes);
    *rep = *this;
    rep->build(*block);

    return rep;
  }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//  (Des,Des,Cre)
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

double SpinAdapted::DesDesCre::redMatrixElement(Csf c1, vector<Csf>& ladder, const SpinBlock* b)
{
  assert( build_pattern == "((DD)(C))" );
  double element = 0.0;
  int I = get_orbs()[0]; 
  int J = get_orbs()[1];
  int K = get_orbs()[2];

  // Must take into account how the 3-index is built from a combination of 2-index and 1-index
  std::vector<SpinQuantum> quantum_ladder = get_quantum_ladder().at("((DD)(C))");
  assert( quantum_ladder.size() == 2 );
  SpinQuantum deltaQuantum12 = quantum_ladder.at(0);
  SpinQuantum deltaQuantum123 = quantum_ladder.at(1);
  deltaQuantum[0] = deltaQuantum123;

  // Spin quantum data for first pair of operators combined
  IrrepSpace sym12 = deltaQuantum12.get_symm();
  int irrep12 = deltaQuantum12.get_symm().getirrep();
  int spin12 = deltaQuantum12.get_s().getirrep();
  // Spin quantum data for total operator
  IrrepSpace sym123 = deltaQuantum123.get_symm();
  int irrep123 = deltaQuantum123.get_symm().getirrep();
  int spin123 = deltaQuantum123.get_s().getirrep();

  TensorOp D1(I,-1); 
  TensorOp D2(J,-1); 
  TensorOp C3(K, 1); 

  // Combine first two operators
//FIXME I=J argument??
  TensorOp DD = D1.product(D2, spin12, irrep12);
  // Combine with third operator
  TensorOp DDC = DD.product(C3, spin123, irrep123);

  for (int i=0; i<ladder.size(); i++)
  {
    int index = 0; double cleb=0.0;
    if (nonZeroTensorComponent(c1, deltaQuantum[0], ladder[i], index, cleb)) {
      std::vector<double> MatElements = calcMatrixElements(c1, DDC, ladder[i]) ;
      element = MatElements[index]/cleb;
      break;
    }
    else
      continue;
  }
  return element;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

boost::shared_ptr<SpinAdapted::SparseMatrix> SpinAdapted::DesDesCre::getworkingrepresentation(const SpinBlock* block)
{
  assert(this->get_initialised());
  if (this->get_built()) {
    return boost::shared_ptr<DesDesCre>(this, boostutils::null_deleter()); // boost::shared_ptr does not own op
  }
  else {
    boost::shared_ptr<SparseMatrix> rep(new DesDesCre);
    *rep = *this;
    rep->build(*block);

    return rep;
  }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------
//  (Des,Cre,Cre)
//-------------------------------------------------------------------------------------------------------------------------------------------------------------

double SpinAdapted::DesCreCre::redMatrixElement(Csf c1, vector<Csf>& ladder, const SpinBlock* b)
{
  assert( build_pattern == "((D)(CC))" );
  double element = 0.0;
  int I = get_orbs()[0]; 
  int J = get_orbs()[1];
  int K = get_orbs()[2];

  // Must take into account how the 3-index is built from a combination of 2-index and 1-index
  std::vector<SpinQuantum> quantum_ladder = get_quantum_ladder().at("((D)(CC))");
  assert( quantum_ladder.size() == 2 );
  SpinQuantum deltaQuantum23 = quantum_ladder.at(0);
  SpinQuantum deltaQuantum123 = quantum_ladder.at(1);
  deltaQuantum[0] = deltaQuantum123;

  // Spin quantum data for first pair of operators combined
  IrrepSpace sym23 = deltaQuantum23.get_symm();
  int irrep23 = deltaQuantum23.get_symm().getirrep();
  int spin23 = deltaQuantum23.get_s().getirrep();
  // Spin quantum data for total operator
  IrrepSpace sym123 = deltaQuantum123.get_symm();
  int irrep123 = deltaQuantum123.get_symm().getirrep();
  int spin123 = deltaQuantum123.get_s().getirrep();

  TensorOp D1(I,-1); 
  TensorOp C2(J, 1); 
  TensorOp C3(K, 1); 

  // Combine first two operators
//FIXME I=J argument??
  TensorOp CC = C2.product(C3, spin23, irrep23);
  // Combine with third operator
  TensorOp DCC = D1.product(CC, spin123, irrep123);

  for (int i=0; i<ladder.size(); i++)
  {
    int index = 0; double cleb=0.0;
    if (nonZeroTensorComponent(c1, deltaQuantum[0], ladder[i], index, cleb)) {
      std::vector<double> MatElements = calcMatrixElements(c1, DCC, ladder[i]) ;
      element = MatElements[index]/cleb;
      break;
    }
    else
      continue;
  }
  return element;
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

boost::shared_ptr<SpinAdapted::SparseMatrix> SpinAdapted::DesCreCre::getworkingrepresentation(const SpinBlock* block)
{
  assert(this->get_initialised());
  if (this->get_built()) {
    return boost::shared_ptr<DesCreCre>(this, boostutils::null_deleter()); // boost::shared_ptr does not own op
  }
  else {
    boost::shared_ptr<SparseMatrix> rep(new DesCreCre);
    *rep = *this;
    rep->build(*block);

    return rep;
  }
}

//-------------------------------------------------------------------------------------------------------------------------------------------------------------

