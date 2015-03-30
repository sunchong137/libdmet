#ifndef WRAPPER_HEADER
#define WRAPPER_HEADER

#ifdef __cplusplus
extern "C" {
#endif

  void ReadInputFromC(char* conf, int outputlevel);
  void readMPSFromDiskAndInitializeStaticVariables(int mpsindex);
  void evaluateOverlapAndHamiltonian(long *occ, int length, double* o, double* h);
  void test();
#ifdef __cplusplus
}
#endif


#endif
