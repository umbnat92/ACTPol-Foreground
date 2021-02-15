import numpy as np 
from scipy import constants
import yaml
import sys

class ReadYaml():
  """
  A simple class that return the yaml file as an object
  """
  def __init__(self,fname):
    with open(fname) as f:
        dataMap = yaml.safe_load(f)

    return self.__dict__.update(dataMap)


class ACTPol_Fgspectra(object):
  def __init__(self,fname):
    self.yaml = ReadYaml(fname)

    self.foreground_template = self.yaml.foregrounds["foreground_template"]

    self.lmax_win = self.yaml.spectra["lmax_win"]
    self.l_bpws = np.arange(2,self.lmax_win+1)

    self.required_cl = self.yaml.spectra["use_spectra"]

    #self.cross_spectra = {"tt":["98x98","98x150","150x150"],"te":["98x98","98x150","150x98","150x150"],
    #                              "ee": ["98x98","98x150","150x150"]}

    self.cross_spectra = self.yaml.spectra["cross_spectra"]

    self.params = self.yaml.params

    self.foregrounds = self.yaml.foregrounds
    self.freqs = self.foregrounds["frequencies"]["nominal"]

    self.expected_params = ["a_tSZ", "a_kSZ", "xi", "a_p", "a_c", "beta_p",
                              "a_s", "a_g","a_gte","a_gee", "a_pste", "a_psee",
                              "T_dd","T_d","n_CIBC"]

    self.fg_params = {k: self.params[k] for k in self.expected_params}

  def initialize_fg(self):
    """
    This function initialize what is needed to compute the foreground following the
    Fortran implementation.

    """
    self.T_cmb = 2.72548

    #Here we load all the normalisation
    #N.B. We need all the effective frequencies as numpy array
    normalisation = self.yaml.foregrounds["normalisation"]
    effetctivenu = self.yaml.foregrounds["frequencies"]

    self.components = self.yaml.foregrounds["components"]
    
    self.feff  = normalisation["nu_0"]
    self.fsz   = np.array(effetctivenu["fsz"])
    self.fdust = np.array(effetctivenu["fdust"])
    self.fsyn = np.array(effetctivenu["fsyn"])

    #These are hardcoded, however one can think to put them into the yaml.
    self.beta_g = 1.5
    self.alpha_s = -0.5
    self.alpha_gs = -1.

    self.sz_func()
    self.planckratiod = self.plankfunctionratio(self.fg_params["T_d"])
    self.planckratiodg = self.plankfunctionratio(self.fg_params["T_dd"])
    self.fluxtempd = self.flux2tempratiod()
    self.fluxtemps = self.flux2tempratios()

    self.read_fg_power()


  def _get_spectra_Froutine(self):
    self.initialize_fg()

    fg_tt_model = self.fg_tt(self.fg_params)
    fg_te_model = self.fg_te(self.fg_params)
    fg_ee_model = self.fg_ee(self.fg_params)

    fg_dict = {}
    component_list = {s: self.components[s] for s in self.required_cl}
    for s in self.required_cl:
      for f in self.cross_spectra[s]:
        fg_dict[s, "all",f] = np.zeros(len(self.l_bpws))
        for comp in component_list[s]:
          if s == 'tt':
            fg_dict[s, comp, f] = fg_tt_model[comp,f]
            fg_dict[s, "all", f] += fg_dict[s,comp, f]
          if s == 'te':
            fg_dict[s, comp, f] = fg_te_model[comp,f]
            fg_dict[s, "all", f] += fg_dict[s,comp, f]
          if s == 'ee':
            fg_dict[s, comp, f] = fg_ee_model[comp,f]
            fg_dict[s, "all", f] += fg_dict[s,comp, f]

    return fg_dict


  def read_fg_power(self):
    self.cltsz = np.loadtxt(self.foreground_template+'cl_tsz_150_bat.dat',usecols=(1),unpack=True)[:self.lmax_win-1]/5.59550
    self.clksz = np.loadtxt(self.foreground_template+'cl_ksz_bat.dat',usecols=(1),unpack=True)[:self.lmax_win-1]/1.51013
    self.clp = self.l_bpws * (self.l_bpws + 1.)/(3000. * 3001.)
    self.clc = np.loadtxt(self.foreground_template+'cib_extra.dat',usecols=(1),unpack=True)[:self.lmax_win-1]
    self.clszcib = np.loadtxt(self.foreground_template+'sz_x_cib_template.dat',usecols=(1),unpack=True)[:self.lmax_win-1]
    self.clgt = (self.l_bpws/500.)**(-0.6)
    self.clgp = (self.l_bpws/500.)**(-0.4)
    self.clsp = (self.l_bpws/500.)**(-0.7)


  def sz_func(self):
    nu = self.fsz * 1e9
    nu0 = self.feff * 1e9
    x = constants.h * nu/(constants.k * self.T_cmb)
    x0 = constants.h * nu0/(constants.k * self.T_cmb)
    
    self.fszcorr = (x*(1/np.tanh(x/2.))-4)
    self.f0 = (x0*(1/np.tanh(x0/2.))-4)


  def plankfunctionratio(self,T_eff):
    nu = self.fdust * 1e9 
    nu0 = self.feff * 1e9
    x = constants.h * nu/(constants.k * T_eff)
    x0 = constants.h * nu0/(constants.k * T_eff)
    
    return (nu/nu0)**3 * (np.exp(x0)-1.)/(np.exp(x)-1.)


  def flux2tempratiod(self):
    nu =  self.fdust * 1e9
    nu0 = self.feff * 1e9
    x = constants.h * nu/(constants.k * self.T_cmb)
    x0 = constants.h * nu0/(constants.k * self.T_cmb)
    
    return (nu0/nu)**4 * (np.exp(x0)/np.exp(x)) * ((np.exp(x)-1.)/(np.exp(x0)-1.))**2


  def flux2tempratios(self):
    nu =  self.fsyn * 1e9
    nu0 = self.feff * 1e9
    x = constants.h * nu/(constants.k * self.T_cmb)
    x0 = constants.h * nu0/(constants.k * self.T_cmb)
    
    return (nu0/nu)**4 * (np.exp(x0)/np.exp(x)) * ((np.exp(x)-1.)/(np.exp(x0)-1.))**2


  def fg_tt(self,fg_params):
    fg_tt_model = {}

    index = self.cross_spectra['tt']
    if "98x98" in index:
      f = "98x98"
      i = 0
      j = 0
      fg_tt_model["cibc",f] = fg_params['a_c'] * self.clc * (self.fdust[i] * self.fdust[j]/(self.feff**2))**fg_params['beta_p']\
                                          * self.planckratiod[i] * self.planckratiod[j] * self.fluxtempd[i] * self.fluxtempd[j]

      fg_tt_model["cibp",f] = fg_params['a_p'] * self.clp * (self.fdust[i] * self.fdust[j]/(self.feff**2))**fg_params['beta_p']\
                                          *  self.planckratiod[i] * self.planckratiod[j] * self.fluxtempd[i] * self.fluxtempd[j]

      fg_tt_model["radio",f] = fg_params['a_s'] * self.clp * (self.fsyn[i] * self.fsyn[j]/(self.feff**2))**self.alpha_s\
                                          * self.fluxtemps[i] * self.fluxtemps[j]

      fg_tt_model["dust",f] = fg_params['a_g'] * self.clgt * (self.fdust[i] * self.fdust[j]/(self.feff**2))**self.beta_g\
                                          * self.planckratiodg[i] * self.planckratiodg[j] * self.fluxtempd[i] * self.fluxtempd[j]

      fg_tt_model["tSZ",f] = fg_params['a_tSZ'] * self.cltsz * self.fszcorr[i] * self.fszcorr[j]/(self.f0 * self.f0)

      fg_tt_model["kSZ",f] = fg_params['a_kSZ'] * self.clksz

      fg_tt_model["tSZxcib",f] = -2. * np.sqrt(fg_params['a_c'] * fg_params['a_tSZ']) * fg_params['xi']\
                                  * self.clszcib * ((self.fdust[i]**fg_params['beta_p'] * self.fszcorr[j]\
                                  * self.planckratiod[i] * self.fluxtempd[i]+self.fdust[j]**fg_params['beta_p'] \
                                  * self.fszcorr[i] * self.planckratiod[j] * self.fluxtempd[j])/(2*self.feff**fg_params['beta_p']*self.f0))

    if "98x150" in index:
      f = "98x150"
      i = 0
      j = 1
      fg_tt_model["cibc",f] = fg_params['a_c'] * self.clc * (self.fdust[i] * self.fdust[j]/(self.feff**2))** fg_params['beta_p']\
                                          * self.planckratiod[i] * self.planckratiod[j] * self.fluxtempd[i] * self.fluxtempd[j]

      fg_tt_model["cibp",f] = fg_params['a_p'] * self.clp * (self.fdust[i] * self.fdust[j]/(self.feff**2))**fg_params['beta_p']\
                                          *  self.planckratiod[i] * self.planckratiod[j] * self.fluxtempd[i] * self.fluxtempd[j]

      fg_tt_model["radio",f] = fg_params['a_s'] * self.clp * (self.fsyn[i] * self.fsyn[j]/(self.feff**2))**self.alpha_s\
                                          * self.fluxtemps[i] * self.fluxtemps[j]

      fg_tt_model["dust",f] = fg_params['a_g'] * self.clgt * (self.fdust[i] * self.fdust[j]/(self.feff**2))**self.beta_g\
                                          * self.planckratiodg[i] * self.planckratiodg[j] * self.fluxtempd[i] * self.fluxtempd[j]

      fg_tt_model["tSZ",f] = fg_params['a_tSZ'] * self.cltsz * self.fszcorr[i] * self.fszcorr[j]/(self.f0 * self.f0)

      fg_tt_model["kSZ",f] = fg_params['a_kSZ'] * self.clksz

      fg_tt_model["tSZxcib",f] = -2. * np.sqrt(fg_params['a_c'] * fg_params['a_tSZ']) * fg_params['xi']\
                                  * self.clszcib * ((self.fdust[i]**fg_params['beta_p'] * self.fszcorr[j]\
                                  * self.planckratiod[i] * self.fluxtempd[i]+self.fdust[j]**fg_params['beta_p'] \
                                  * self.fszcorr[i] * self.planckratiod[j] * self.fluxtempd[j])/(2*self.feff**fg_params['beta_p']*self.f0))


    if "150x150" in index:
      f = "150x150"
      i = 1
      j = 1
      fg_tt_model["cibc",f] = fg_params['a_c'] * self.clc * (self.fdust[i] * self.fdust[j]/(self.feff**2))** fg_params['beta_p']\
                                          * self.planckratiod[i] * self.planckratiod[j] * self.fluxtempd[i] * self.fluxtempd[j]

      fg_tt_model["cibp",f] = fg_params['a_p'] * self.clp * (self.fdust[i] * self.fdust[j]/(self.feff**2))**fg_params['beta_p']\
                                          *  self.planckratiod[i] * self.planckratiod[j] * self.fluxtempd[i] * self.fluxtempd[j]

      fg_tt_model["radio",f] = fg_params['a_s'] * self.clp * (self.fsyn[i] * self.fsyn[j]/(self.feff**2))**self.alpha_s\
                                          * self.fluxtemps[i] * self.fluxtemps[j]

      fg_tt_model["dust",f] = fg_params['a_g'] * self.clgt * (self.fdust[i] * self.fdust[j]/(self.feff**2))**self.beta_g\
                                          * self.planckratiodg[i] * self.planckratiodg[j] * self.fluxtempd[i] * self.fluxtempd[j]

      fg_tt_model["tSZ",f] = fg_params['a_tSZ'] * self.cltsz * self.fszcorr[i] * self.fszcorr[j]/(self.f0 * self.f0)

      fg_tt_model["kSZ",f] = fg_params['a_kSZ'] * self.clksz

      fg_tt_model["tSZxcib",f] = -2. * np.sqrt(fg_params['a_c'] * fg_params['a_tSZ']) * fg_params['xi']\
                                  * self.clszcib * ((self.fdust[i]**fg_params['beta_p'] * self.fszcorr[j]\
                                  * self.planckratiod[i] * self.fluxtempd[i]+self.fdust[j]**fg_params['beta_p'] \
                                  * self.fszcorr[i] * self.planckratiod[j] * self.fluxtempd[j])/(2*self.feff**fg_params['beta_p']*self.f0))

    return fg_tt_model


  def fg_ee(self,fg_params):
    fg_ee_model = {}

    index = self.cross_spectra['ee']
    if "98x98" in index:
      f = "98x98"
      i = 0
      j = 0
      
      fg_ee_model["radio",f] = fg_params['a_psee'] * self.clp *((self.fsyn[i] * self.fsyn[j]/self.feff**2)**self.alpha_s)\
                                    * self.fluxtemps[i] * self.fluxtemps[j]

      fg_ee_model["dust",f] = fg_params['a_gee'] * self.clgp * ((self.fdust[i] * self.fdust[j]/self.feff**2)**self.beta_g)\
                                    * self.planckratiodg[i] * self.planckratiodg[j] * self.fluxtempd[i] * self.fluxtempd[j]


    if "98x150" in index:
      f = "98x150"
      i = 0
      j = 1
      
      fg_ee_model["radio",f] = fg_params['a_psee'] * self.clp *((self.fsyn[i] * self.fsyn[j]/self.feff**2)**self.alpha_s)\
                                    * self.fluxtemps[i] * self.fluxtemps[j]

      fg_ee_model["dust",f] = fg_params['a_gee'] * self.clgp * ((self.fdust[i] * self.fdust[j]/self.feff**2)**self.beta_g)\
                                    * self.planckratiodg[i] * self.planckratiodg[j] * self.fluxtempd[i] * self.fluxtempd[j]


    if "150x150" in index:
      f = "150x150"
      i = 1
      j = 1
      
      fg_ee_model["radio",f] = fg_params['a_psee'] * self.clp *((self.fsyn[i] * self.fsyn[j]/self.feff**2)**self.alpha_s)\
                                    * self.fluxtemps[i] * self.fluxtemps[j]

      fg_ee_model["dust",f] = fg_params['a_gee'] * self.clgp * ((self.fdust[i] * self.fdust[j]/self.feff**2)**self.beta_g)\
                                    * self.planckratiodg[i] * self.planckratiodg[j] * self.fluxtempd[i] * self.fluxtempd[j]


    return fg_ee_model

  def fg_te(self,fg_params):
    fg_te_model = {}

    index = self.cross_spectra['te']
    if "98x98" in index:
      f = "98x98"
      i = 0
      j = 0

      fg_te_model["radio",f] = fg_params['a_pste'] * self.clp * ((self.fsyn[i] * self.fsyn[j]/self.feff**2)**self.alpha_s)\
                                    * self.fluxtemps[i] * self.fluxtemps[j]

      fg_te_model["dust",f] = fg_params['a_gte'] * self.clgp * ((self.fdust[i] * self.fdust[j]/self.feff**2)**self.beta_g)\
                                    * self.planckratiodg[i] * self.planckratiodg[j] * self.fluxtempd[i] * self.fluxtempd[j]

    if "98x150" in index:
      f = "98x150"
      i = 0
      j = 1

      fg_te_model["radio",f] = fg_params['a_pste'] * self.clp * ((self.fsyn[i] * self.fsyn[j]/self.feff**2)**self.alpha_s)\
                                    * self.fluxtemps[i] * self.fluxtemps[j]

      fg_te_model["dust",f] = fg_params['a_gte'] * self.clgp * ((self.fdust[i] * self.fdust[j]/self.feff**2)**self.beta_g)\
                                    * self.planckratiodg[i] * self.planckratiodg[j] * self.fluxtempd[i] * self.fluxtempd[j]


    if "150x98" in index:
      f = "150x98"
      i = 1
      j = 0

      fg_te_model["radio",f] = fg_params['a_pste'] * self.clp * ((self.fsyn[i] * self.fsyn[j]/self.feff**2)**self.alpha_s)\
                                    * self.fluxtemps[i] * self.fluxtemps[j]

      fg_te_model["dust",f] = fg_params['a_gte'] * self.clgp * ((self.fdust[i] * self.fdust[j]/self.feff**2)**self.beta_g)\
                                    * self.planckratiodg[i] * self.planckratiodg[j] * self.fluxtempd[i] * self.fluxtempd[j]


    if "150x150" in index:
      f = "150x150"
      i = 1
      j = 1

      fg_te_model["radio",f] = fg_params['a_pste'] * self.clp * ((self.fsyn[i] * self.fsyn[j]/self.feff**2)**self.alpha_s)\
                                    * self.fluxtemps[i] * self.fluxtemps[j]

      fg_te_model["dust",f] = fg_params['a_gte'] * self.clgp * ((self.fdust[i] * self.fdust[j]/self.feff**2)**self.beta_g)\
                                    * self.planckratiodg[i] * self.planckratiodg[j] * self.fluxtempd[i] * self.fluxtempd[j]

    return fg_te_model

  def _get_spectra_fgspectra(self):
    fgpar = {}

    return get_foreground_model(fg_params=self.fg_params,
                                    fg_model=self.foregrounds,
                                    frequencies=self.freqs,
                                    ell=self.l_bpws,
                                    requested_cls=self.required_cl)




# Standalone function to return the foreground model
# given the nuisance parameters
def get_foreground_model(fg_params, fg_model,
                       frequencies, ell,
                       requested_cls=["tt", "te", "ee"]):
  
  normalisation = fg_model["normalisation"]
  nu_0 = normalisation["nu_0"]
  ell_0 = normalisation["ell_0"]

  freq_eff = fg_model["frequencies"]
  
  fdust = freq_eff["fdust"]
  fsz = freq_eff["fsz"]
  fsyn = freq_eff["fsyn"]

  external_cl = fg_model["foreground_template"]
  cibc_file = external_cl+'cib_extra.dat'
  tSZxcib_file = external_cl+'sz_x_cib_template.dat'

  clszcib = np.loadtxt(tSZxcib_file,usecols=(1),unpack=True)[:ell.max()-1]  

  ell_clp = ell*(ell+1.)
  ell_0clp = 3000.*3001.


  T_CMB = 2.72548

  from fgspectra import cross as fgc
  from fgspectra import frequency as fgf
  from fgspectra import power as fgp

  cibc = fgc.FactorizedCrossSpectrum(fgf.CIB(), fgp.PowerSpectrumFromFile(cibc_file))
  cibp = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
  radio = fgc.FactorizedCrossSpectrum(fgf.PowerLaw(), fgp.PowerLaw())
  dust = fgc.FactorizedCrossSpectrum(fgf.ModifiedBlackBody(), fgp.PowerLaw())
  tsz = fgc.FactorizedCrossSpectrum(fgf.ThermalSZ(), fgp.tSZ_150_bat())
  ksz = fgc.FactorizedCrossSpectrum(fgf.ConstantSED(), fgp.kSZ_bat())
  tSZ_and_CIB = fgc.CorrelatedFactorizedCrossSpectrum(fgf.Join(fgf.ThermalSZ(), fgf.CIB()),fgp.SZxCIB_Addison2012())

  # Make sure to pass a numpy array to fgspectra
  if not isinstance(frequencies, np.ndarray):
      frequencies = np.array(frequencies)

  if not isinstance(fdust, np.ndarray):
      fdust = np.array(fdust)


  if not isinstance(fsz, np.ndarray):
      fsz = np.array(fsz)

  if not isinstance(fsyn, np.ndarray):
      fsyn = np.array(fsyn)

  fszcorr,f0 = sz_func(fsz,nu_0)
  planckratiod = plankfunctionratio(fdust,nu_0,fg_params["T_d"])
  fluxtempd = flux2tempratiod(fdust,nu_0)

  model = {}
  
  #TT Foreground
  model["tt", "cibc"] = fg_params["a_c"] * cibc(
      {"nu": fdust, "nu_0": nu_0,
       "temp": fg_params["T_d"], "beta": fg_params["beta_p"]},
       {'ell':ell, 'ell_0':ell_0})

  model["tt", "cibp"] = fg_params["a_p"] * cibp(
      {"nu": fdust, "nu_0": nu_0,
       "temp": fg_params["T_d"], "beta": fg_params["beta_p"]},
      {"ell": ell_clp, "ell_0": ell_0clp, "alpha": 1})

  model["tt", "tSZ"] = fg_params["a_tSZ"] * tsz(
      {"nu": fsz, "nu_0": nu_0},
      {"ell": ell, "ell_0": ell_0})

  model["tt", "kSZ"] = fg_params["a_kSZ"] * ksz(
      {"nu": (fsz/fsz)},
      {"ell": ell, "ell_0": ell_0})
  

  model["ee", "radio"] = fg_params["a_psee"] * radio(
      {"nu": fsyn, "nu_0": nu_0, "beta": -0.5 -2.},
      {"ell": ell_clp, "ell_0": ell_0clp,"alpha":1})
  
  model["te", "radio"] = fg_params["a_pste"] * radio(
      {"nu": fsyn, "nu_0": nu_0, "beta": -0.5 -2.},
      {"ell": ell_clp, "ell_0": ell_0clp,"alpha":1.})

  model["tt", "dust"] = fg_params["a_g"] * dust(
          {"nu": fdust, "nu_0": nu_0,
          "temp": fg_params["T_dd"], "beta": 1.5},
          {"ell": ell, "ell_0": 500., "alpha": -0.6})

  model["tt", "radio"] = fg_params["a_s"] * radio(
      {"nu": fsyn, "nu_0": nu_0, "beta": -0.5 -2.},
      {"ell": ell_clp, "ell_0": ell_0clp,"alpha":1})      
  
  model["ee", "dust"] = fg_params["a_gee"] * dust(
      {"nu": fdust, "nu_0": nu_0,
      "temp": fg_params["T_dd"], "beta": 1.5},
      {"ell": ell, "ell_0": 500., "alpha": -0.4})
  
  model["te", "dust"] = fg_params["a_gte"] * dust(
      {"nu": fdust, "nu_0": nu_0,
      "temp": fg_params["T_dd"], "beta": 1.5},
      {"ell": ell, "ell_0": 500., "alpha": -0.4})

  components = fg_model["components"]
  component_list = {s: components[s] for s in requested_cls}
  fg_dict = {}
  for c1, f1 in enumerate(frequencies):
    for c2, f2 in enumerate(frequencies):
      for s in requested_cls:
        fg_dict[s, "all", f1, f2] = np.zeros(len(ell))
        for comp in component_list[s]:
          if comp == "tSZxcib":
            fg_dict[s, comp, f1, f2] = -2. * np.sqrt(fg_params['a_c'] * fg_params['a_tSZ']) * fg_params['xi']\
                                  * clszcib * ((fdust[c1]**fg_params['beta_p'] * fszcorr[c2]\
                                  * planckratiod[c1] * fluxtempd[c1]+fdust[c2]**fg_params['beta_p'] \
                                  * fszcorr[c1] * planckratiod[c2] * fluxtempd[c2])/(2 * nu_0**fg_params['beta_p'] * f0))
          else:
            fg_dict[s, comp, f1, f2] = model[s, comp][c1, c2]
            fg_dict[s, "all", f1, f2] += fg_dict[s, comp, f1, f2]


  return fg_dict

def sz_func(fsz,feff):
  T_cmb = 2.72548
  nu = fsz * 1e9
  nu0 = feff * 1e9
  x = constants.h * nu/(constants.k * T_cmb)
  x0 = constants.h * nu0/(constants.k * T_cmb)
  
  fszcorr = (x*(1/np.tanh(x/2.))-4)
  f0 = (x0*(1/np.tanh(x0/2.))-4)
  return fszcorr,f0
  

def plankfunctionratio(fdust,feff,T_eff):
  nu = fdust * 1e9 
  nu0 = feff * 1e9
  x = constants.h * nu/(constants.k * T_eff)
  x0 = constants.h * nu0/(constants.k * T_eff)
  
  return (nu/nu0)**3 * (np.exp(x0)-1.)/(np.exp(x)-1.)


def flux2tempratiod(fdust,feff):
  T_cmb = 2.72548
  nu =  fdust * 1e9
  nu0 = feff * 1e9
  x = constants.h * nu/(constants.k * T_cmb)
  x0 = constants.h * nu0/(constants.k * T_cmb)
  
  return (nu0/nu)**4 * (np.exp(x0)/np.exp(x)) * ((np.exp(x)-1.)/(np.exp(x0)-1.))**2

  