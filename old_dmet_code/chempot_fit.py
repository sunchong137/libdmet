import numpy as np
from math import sqrt, copysign

# functions related to fitting mu

def num_diff(func, center, h, option, info = None):
  if option == "3":
    trial = np.linspace(-1, 1, 3) * h + center
  elif option == "5":
    trial = np.linspace(-2, 2, 5) * h + center
  elif option == "2":
    trial = np.linspace(-1, 1, 2) * h + center
  elif option == "2to3":
    assert(info is not None)
    trial = [center]
  else:
    raise Exception("Undefined method")
  
  v = map(func, trial)

  if option == "3":
    f0 = v[1]
    f1 = (v[2] - v[0]) / (2.*h)
    f2 = (v[0]-2.*v[1]+v[2]) / h**2
  elif option == "5":
    f0 = v[2]
    f1 = (v[0]-8.*v[1]+8.*v[3]-v[4]) / (12. * h)
    f2 = (-v[0]+16.*v[1]-30.*v[2]+16.*v[3]-v[4]) / (12. * h**2)
  elif option == "2":
    f0 = (v[0] + v[1]) / 2
    f1 = (v[1] - v[0]) / (2.*h)
    f2 = None
  elif option == "2to3":
    f0 = v[0]
    f1 = None
    f2 = (info-v[0])*2./h**2
  return f0, f1, f2

def ChemPotDeriSearchEmb(fn, mu0, n0, step = 0.05, verbose = 0):
  if verbose > 1:
    print "Derivative Search:"
  n, n1, info = num_diff(fn, mu0, step, "2")
  # try first order derivative
  if abs(n1) > 1e-7:
    dmu = (n0-n) / n1
  else:
    dmu = copysign(0.1, n0-n)

  if abs(dmu) > 0.3:
    dmu = copysign(0.3, dmu)
  #elif abs(dmu) < 0.15:
  #  n, info, n2 = num_diff(fn, mu0, step, "2to3", n)
  #  discriminant = n1**2+2*n2*(n0-n)
  #  if discriminant > 0:
  #    # then use quadratic approximation
  #    dmu = (sqrt(discriminant)-abs(n1)) / n2 * copysign(1., n1)
  
  return dmu

def ChemPotDeriSearchMF(fn, mu0, n0, max_iter = 10, step = 0.2, ntol = 1e-3, verbose = 0):
  if verbose > 1:
    print "Derivative Search:"
  mu = mu0
  mus = [mu]
  ns = []
  for i in range(max_iter):
    n, n1, n2 = num_diff(fn, mu, step, "3")
    ns.append(n)
    if verbose > 2:
      print "Iter %2d mu = %20.12f nelec = %20.12f" % (i, mu, n)
    # first convergence criteria
    if abs(n-n0) < ntol:
      return mu - mu0, None, None, True # converged
    # second criteria: section search condition
    if i > 0 and (ns[-2]-n0) * (ns[-1]-n0) < 0:
      mu_int = mus[-2:]
      n_int = ns[-2:]
      if mu_int[0] > mu_int[1]:
        mu_int.reverse()
        n_int.reverse()
      return None, mu_int, n_int, False # need to do section search

    # compute next mu
    if abs(n1) > 1e-6:
      dmu = (n0-n) / n1
    else:
      dmu = copysign(1., n0-n)
    if abs(dmu) > 1.:
      dmu = copysign(1., dmu)
    elif abs(dmu) < 0.2:
      discriminant = n1**2+2*n2*(n0-n)
      if discriminant > 0:
        dmu = (sqrt(discriminant)-abs(n1)) / n2 * copysign(1., n1)
    mu += dmu
    mus.append(mu)
    
  return mu-mu0, None, None, False # not converged, and cannot do section search 

def section_guess(x, y, y0, method = "golden"):
  if method == "golden":
    phi = [(3.-sqrt(5.).real)*0.5, (sqrt(5.).real-1.)*0.5]
    if abs(y[0] - y0) > abs(y[1] - y0):
      return phi[0] * x[0] + phi[1] * x[1]
    else:
      return phi[1] * x[0] + phi[0] * x[1]
  elif method == "linear":
    phi = np.array([abs(y[1] - y0), abs(y[0] - y0)])
    phi /= np.sum(phi)
    return phi[0] * x[0] + phi[1] * x[1]
  elif method == "quad":
    phi = np.array([abs(y[1] - y0) ** 0.5, abs(y[0] - y0) ** 0.5])
    phi /= np.sum(phi)
    return phi[0] * x[0] + phi[1] * x[1]
  elif isinstance(method, float):
    phi = np.array([abs(y[1] - y0) ** method, abs(y[0] - y0) ** method])
    phi /= np.sum(phi)
    return phi[0] * x[0] + phi[1] * x[1]
  else:
    raise Exception("Invalid section search method.")

def ChemPotSectionSearch(fn, mus, ns, n0, mutol = 1e-5, ntol = 1e-8, verbose = 0):
  assert(mus[1] > mus[0])
  if verbose > 1:
    print "Section Search"
  iter = 0
  while mus[1] - mus[0] > mutol*2 and ns[1] - ns[0] > ntol*2:
    mu = section_guess(mus, ns, n0, method = "golden")
    n = fn(mu)
    if verbose > 2:
      print "Iter %2d mu = %20.12f nelec = %20.12f" % (iter, mu, n)
    if (n-n0) * (ns[0]-n0) > 0.:
      mus[0], ns[0] = mu, n
    else:
      mus[1], ns[1] = mu, n
    iter += 1
  mu = section_guess(mus, ns, n0, "linear")
  return mu

def Fit_ChemicalPotential_MF(fn, mu0, n0, inp_fit, verbose = 0):
  if verbose > 1:
    print "\n********** Fitting Chemical Potential (HFB) *************\n"
    print "Target Nelec per cell and spin = %20.12f" % n0
  # because of the property that n is monotonic wrt mu, and kinda piecewise function, 
  # we first use derivatives to do fitting, after we have a range, use
  # section search to refine
  dmu, mus, ns, conv = ChemPotDeriSearchMF(fn, mu0, n0, max_iter = 15, step = 0.2, ntol = 1e-4, verbose = verbose)
  if conv:
    print "Derivative search converged"
  elif mus is None:
    print "Warning: Mean-field chemical potential fitting not converged!"
  else:
    dmu = ChemPotSectionSearch(fn, mus, ns, n0, mutol = 1e-6, ntol = 1e-8, verbose = verbose) - mu0
  
  if verbose > 1:
    print
  
  return dmu

def Fit_ChemicalPotential_Emb(fn, mu0, n0, inp_fit, verbose = 0):
  if verbose > 1:
    print "\n******** Fitting Chemical Potential (Embedding) *********\n"
    print "Target Nelec per cell and spin = %20.12f" % n0
    print "initial chemical potential = %20.12f" % mu0

  dmu = ChemPotDeriSearchEmb(fn, mu0, n0, step = 0.03, verbose = verbose)
  if verbose > 1:
    print
  return dmu


def Fit_ChemicalPotential_Emb_special(fn, mu0, n0, na, inp_fit, verbose = 0):
  if verbose > 1:
    print "\n******** Fitting Chemical Potential (Embedding) *********\n"
    print "Target Nelec per cell and spin = %20.12f" % n0
    print "Derivative Search:"
  
  step = 0.01

  if na > n0:
    step *= -1
  
  nb = fn(mu0+step*2)
  
  n = (na+nb)/2
  n1 = (nb-na)/(step*2)
  
  if abs(n1) > 1e-7:
    dmu = (n0-n) / n1
  else:
    dmu = copysign(0.02, n0-n)
  if abs(dmu) > 0.3:
    dmu = copysign(0.3, dmu)
 
  return dmu+step

