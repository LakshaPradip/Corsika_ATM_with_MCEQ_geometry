import sys
import numpy as np
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from os.path import join
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from time import time
from scipy.interpolate import UnivariateSpline

h_obs = 33.0e3 #height of detector from surface of earth in m
r_E = 6391.e3 #Radius of the Earth in m
h_atm = 112.8e3 #top of the atmosphere in m
max_density = 0.001225
n_steps = 5000

def theta_rad(theta):
    """Converts :math:`\\theta` from rad to degrees.
    """
    return np.deg2rad(theta)

def planar_rho_inv(X , cos_theta, a, b, c, t, hl):
    res = 0.0
    x_v = X * cos_theta
    layer = 0
    for i in range(5):
        if not (x_v >= t[i]):
            layer = i
            
        if layer == 4:
            res = c[4]/b[4]
            
        else:
            res = c[layer]/(x_v - a[layer])
            
    return res
    
def corsika_get_density(h_cm, a, b, c, t, hl):
    res = 0.0
    layer = 0
    for i in range(5):
        if not (h_cm <= hl[i]):
            layer = i
    
    if layer == 4:
        res = b[4]/c[4]
    
    else:
        res = b[layer] / c[layer] * np.exp(-h_cm/c[layer])
        
    return res
    
def corsika_get_m_overburden(h_cm, a, b, c, t, hl):
    res = 0.0
    layer = 0
    for i in range(5):
        if not (h_cm <= hl[i]):
            layer = i
    
    if layer==4:
        res = a[4] - b[4] / c[4] * h_cm
    else:
        res = a[layer] + b[layer] * np.exp(-h_cm/c[layer])
        
    return res
        
def rho_lipari(h):
    if h>11.0:
        rho_lipari = 2.054e-3 * np.exp(-h/6.344)
    else:
        rho_lipari = 1.210e-10 * (44.33-h)**4.253
    return rho_lipari


class EarthGeometry(object):
    
    def __init__(self):
        
        self.h_obs = h_obs * 1e2 
        self.h_atm = h_atm * 1e2
        self.r_E = r_E * 1e2
        self.r_top = self.r_E + self.h_atm
        self.r_obs = self.r_E + self.h_obs
        
    def _A_1(self, theta):
        return self.r_obs * np.cos(theta)
    
    def _A_2(self, theta):
        return self.r_obs * np.sin(theta)
        
    def l(self, theta):
        r"""Returns path length in [cm] for given zenith
        angle :math:`\theta` [rad].
        """
        return (np.sqrt(self.r_top**2 - self._A_2(theta)**2) -
                self._A_1(theta))

    def cos_th_star(self, theta):
        r"""Returns the zenith angle at atmospheric boarder
        :math:`\cos(\theta^*)` in [rad] as a function of zenith at detector.
        """
        return (self._A_1(theta) + self.l(theta)) / self.r_top

    def h(self, dl, theta):
        r"""Height above surface at distance :math:`dl` counted from the beginning
        of path :math:`l(\theta)` in cm.
        """
        return np.sqrt(
            self._A_2(theta)**2 +
            (self._A_1(theta) + self.l(theta) - dl)**2) - self.r_E

    def delta_l(self, h, theta):
        r"""Distance :math:`dl` covered along path :math:`l(\theta)`
        as a function of current height. Inverse to :func:`h`.
        """
        return (self._A_1(theta) + self.l(theta) -
                np.sqrt((h + self.r_E)**2 - self._A_2(theta)**2))
                

class EarthsAtmosphere(with_metaclass(ABCMeta)):
    def __init__(self, *args, **kwargs):
        from Corsika_atm import EarthGeometry

        self.geom = kwargs.pop("geometry", EarthGeometry())
        self.thrad = None
        self.theta_deg = None
        self._max_den = max_density
        self.max_theta = 90.0
        self.location = None
        self.season = None
        
    @abstractmethod
    def get_density(self, h_cm):
        """Abstract method which implementation  should return the density in g/cm**3.

        Args:
           h_cm (float):  height in cm

        Returns:
           float: density in g/cm**3

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError("Base class called.")

    def calculate_density_spline(self, n_steps = n_steps):
        """Calculates and stores a spline of :math:`\\rho(X)`.

        Args:
          n_steps (int, optional): number of :math:`X` values
                                   to use for interpolation

        Raises:
            Exception: if :func:`set_theta` was not called before.
        """


        thrad = self.thrad
        path_length = self.geom.l(thrad)
        vec_rho_l = np.vectorize(
            lambda delta_l: self.get_density(self.geom.h(delta_l, thrad))
        )
        dl_vec = np.linspace(0, path_length, n_steps)

        now = time()

        # Calculate integral for each depth point
        X_int = cumulative_trapezoid(vec_rho_l(dl_vec), dl_vec)  #
        dl_vec = dl_vec[1:]


        # Save depth value at h_obs
        self._max_X = X_int[-1]
        self._max_den = self.get_density(self.geom.h(0, thrad))

        # Interpolate with bi-splines without smoothing
        h_intp = [self.geom.h(dl, thrad) for dl in reversed(dl_vec[1:])]
        X_intp = [X for X in reversed(X_int[1:])]

        self._s_h2X = interp1d(h_intp, np.log(X_intp), kind='cubic', fill_value='extrapolate')
        self._s_X2rho = interp1d(X_int, vec_rho_l(dl_vec), kind='cubic', fill_value='extrapolate')
        self._s_lX2h = interp1d(np.log(X_intp)[::-1], h_intp[::-1], kind='cubic', fill_value='extrapolate')
        #print("printed x2rho",self._s_X2rho)
        return X_int
    @property
    def max_X(self):
        """Depth at altitude 0."""
        if not hasattr(self, "_max_X"):
            self.set_theta(0)
        return self._max_X

    @property
    def max_den(self):
        """Density at altitude 0."""
        if not hasattr(self, "_max_den"):
            self.set_theta(0)
        return self._max_den

    @property
    def s_h2X(self):
        """Spline for conversion from altitude to depth."""
        if not hasattr(self, "_s_h2X"):
            self.set_theta(0)
        return self._s_h2X

    @property
    def s_X2rho(self):
        """Spline for conversion from depth to density."""
        if not hasattr(self, "_s_X2rho"):
            self.set_theta(0)
        return self._s_X2rho

    @property
    def s_lX2h(self):
        """Spline for conversion from depth to altitude."""
        if not hasattr(self, "_s_lX2h"):
            self.set_theta(0)
        return self._s_lX2h

    def set_theta(self, theta_deg):
        """Configures geometry and initiates spline calculation for
        :math:`\\rho(X)`.

        If the option 'use_atm_cache' is enabled in the config, the
        function will check, if a corresponding spline is available
        in the cache and use it. Otherwise it will call
        :func:`calculate_density_spline`,  make the function
        :func:`r_X2rho` available to the core code and store the spline
        in the cache.

        Args:
          theta_deg (float): zenith angle :math:`\\theta` at detector
        """
        #if theta_deg < 0.0 or theta_deg > self.max_theta:
            #raise Exception("Zenith angle not in allowed range.")

        self.thrad = theta_rad(theta_deg)
        self.theta_deg = theta_deg
        self.calculate_density_spline()
        
        
    def r_X2rho(self, X):
        """Returns the inverse density :math:`\\frac{1}{\\rho}(X)`.

        The spline `s_X2rho` is used, which was calculated or retrieved
        from cache during the :func:`set_theta` call.

        Args:
           X (float):  slant depth in g/cm**2

        Returns:
           float: :math:`1/\\rho` in cm**3/g

        """
        return 1.0 / self.s_X2rho(X)

    def h2X(self, h):
        """Returns the depth along path as function of height above
        surface.

        The spline `s_X2rho` is used, which was calculated or retrieved
        from cache during the :func:`set_theta` call.

        Args:
           h (float):  vertical height above surface in cm

        Returns:
           float: X  slant depth in g/cm**2

        """
        return np.exp(self.s_h2X(h))

    def X2h(self, X):
        """Returns the height above surface as a function of slant depth
        for currently selected zenith angle.

        The spline `s_lX2h` is used, which was calculated or retrieved
        from cache during the :func:`set_theta` call.

        Args:
           X (float):  slant depth in g/cm**2

        Returns:
           float h:  height above surface in cm

        """
        return self.s_lX2h(np.log(X))
        
    def hVx_plot(self, theta_deg1):
        
        thrad = theta_rad(theta_deg1)
        path_length1 = self.geom.l(thrad)
        dl_vec1 = np.linspace(0, path_length1, n_steps-1)
        h_vec = self.geom.h(dl_vec1, thrad)
        vec_rho_l = np.vectorize(
            lambda delta_l: self.get_density(self.geom.h(delta_l, thrad))
        )
        dl_vec = np.linspace(0, path_length1, n_steps)
        X_int = cumulative_trapezoid(vec_rho_l(dl_vec), dl_vec)
        #print("max X", X_int[-1])
        #print("max height", h_vec[0])
        return(h_vec, X_int)  
        
    def hVrho_plot(self, theta_deg):
        
        thrad = theta_rad(theta_deg)
        path_length = self.geom.l(thrad)
        dl_vec = np.linspace(0, path_length, n_steps-1)
        h_vec = self.geom.h(dl_vec, thrad)
        vec_rho_l = np.vectorize(
            lambda delta_l: self.get_density(self.geom.h(delta_l, thrad))
        )
        dl_vec = np.linspace(0, path_length, n_steps)
        X_int = cumulative_trapezoid(vec_rho_l(dl_vec), dl_vec)
        rho_val = self.X2rho(X_int)
        return(h_vec, rho_val)  
        
    def X2rho(self, X):
        """Returns the density :math:`\\rho(X)`.

        The spline `s_X2rho` is used, which was calculated or retrieved
        from cache during the :func:`set_theta` call.

        Args:
           X (float):  slant depth in g/cm**2

        Returns:
           float: :math:`\\rho` in cm**3/g

        """
        return self.s_X2rho(X)

    def moliere_air(self, h_cm):
        """Returns the Moliere unit of air for US standard atmosphere."""

        return 9.3 / (self.get_density(h_cm) * 100.0)

    def nref_rel_air(self, h_cm):
        """Returns the refractive index - 1 in air (density parametrization
        as in CORSIKA).
        """

        return 0.000283 * self.get_density(h_cm) / self.get_density(0)

    def gamma_cherenkov_air(self, h_cm):
        """Returns the Lorentz factor gamma of Cherenkov threshold in air (MeV)."""

        nrel = self.nref_rel_air(h_cm)
        return (1.0 + nrel) / np.sqrt(2.0 * nrel + nrel**2)

    def theta_cherenkov_air(self, h_cm):
        """Returns the Cherenkov angle in air (degrees)."""

        return np.arccos(1.0 / (1.0 + self.nref_rel_air(h_cm))) * 180.0 / np.pi


class CorsikaAtmosphere(EarthsAtmosphere):
    """Class, holding the parameters of a Linsley type parameterization
    similar to the Air-Shower Monte Carlo
    `CORSIKA <https://web.ikp.kit.edu/corsika/>`_.

    The parameters pre-defined parameters are taken from the CORSIKA
    manual. If new sets of parameters are added to :func:`init_parameters`,
    the array _thickl can be calculated using :func:`calc_thickl` .

    Attributes:
      _atm_param (numpy.array): (5x5) Stores 5 atmospheric parameters
                                _aatm, _batm, _catm, _thickl, _hlay
                                for each of the 5 layers

    Args:
      location (str): see :func:`init_parameters`
      season (str,optional): see :func:`init_parameters`
    """

    _atm_param = None

    def __init__(self, location, season=None):
        cka_atmospheres = [
            ("USStd", None),
            ("BK_USStd", None),
            ("Karlsruhe", None),
            ("ANTARES/KM3NeT-ORCA", "Summer"),
            ("ANTARES/KM3NeT-ORCA", "Winter"),
            ("KM3NeT-ARCA", "Summer"),
            ("KM3NeT-ARCA", "Winter"),
            ("KM3NeT", None),
            ("SouthPole", "December"),
            ("PL_SouthPole", "January"),
            ("PL_SouthPole", "August"),
        ]
        assert (
            location,
            season,
        ) in cka_atmospheres, "{0}/{1} not available for CorsikaAtmsophere".format(
            location, season
        )
        self.init_parameters(location, season)
        EarthsAtmosphere.__init__(self)

    def init_parameters(self, location, season):
        """Initializes :attr:`_atm_param`. Parameters from ANTARES/KM3NET
        are based on the work of T. Heid
        (`see this issue <https://github.com/afedynitch/MCEq/issues/12>`_)

        +---------------------+-------------------+------------------------------+
        | location            | CORSIKA Table     | Description/season           |
        +=====================+===================+==============================+
        | "USStd"             |         23        |  US Standard atmosphere      |
        +---------------------+-------------------+------------------------------+
        | "BK_USStd"          |         37        |  Bianca Keilhauer's USStd    |
        +---------------------+-------------------+------------------------------+
        | "Karlsruhe"         |         24        |  AT115 / Karlsruhe           |
        +---------------------+-------------------+------------------------------+
        | "SouthPole"         |      26 and 28    |  MSIS-90-E for Dec and June  |
        +---------------------+-------------------+------------------------------+
        |"PL_SouthPole"       |      29 and 30    |  P. Lipari's  Jan and Aug    |
        +---------------------+-------------------+------------------------------+
        |"ANTARES/KM3NeT-ORCA"|    NA             |  PhD T. Heid                 |
        +---------------------+-------------------+------------------------------+
        | "KM3NeT-ARCA"       |    NA             |  PhD T. Heid                 |
        +---------------------+-------------------+------------------------------+


        Args:
          location (str): see table
          season (str, optional): choice of season for supported locations

        Raises:
          Exception: if parameter set not available
        """
        _aatm, _batm, _catm, _thickl, _hlay = None, None, None, None, None
        if location == "USStd":
            _aatm = np.array([-186.5562, -94.919, 0.61289, 0.0, 0.01128292])
            _batm = np.array([1222.6562, 1144.9069, 1305.5948, 540.1778, 1.0])
            _catm = np.array([994186.38, 878153.55, 636143.04, 772170.0, 1.0e9])
            _thickl = np.array(
                [1036.102549, 631.100309, 271.700230, 3.039494, 0.001280]
            )
            _hlay = np.array([0, 4.0e5, 1.0e6, 4.0e6, 1.0e7])
        elif location == "BK_USStd":
            _aatm = np.array(
                [-149.801663, -57.932486, 0.63631894, 4.3545369e-4, 0.01128292]
            )
            _batm = np.array([1183.6071, 1143.0425, 1322.9748, 655.69307, 1.0])
            _catm = np.array([954248.34, 800005.34, 629568.93, 737521.77, 1.0e9])
            _thickl = np.array(
                [1033.804941, 418.557770, 216.981635, 4.344861, 0.001280]
            )
            _hlay = np.array([0.0, 7.0e5, 1.14e6, 3.7e6, 1.0e7])
        elif location == "Karlsruhe":
            _aatm = np.array([-118.1277, -154.258, 0.4191499, 5.4094056e-4, 0.01128292])
            _batm = np.array([1173.9861, 1205.7625, 1386.7807, 555.8935, 1.0])
            _catm = np.array([919546.0, 963267.92, 614315.0, 739059.6, 1.0e9])
            _thickl = np.array(
                [1055.858707, 641.755364, 272.720974, 2.480633, 0.001280]
            )
            _hlay = np.array([0.0, 4.0e5, 1.0e6, 4.0e6, 1.0e7])
        elif location == "KM3NeT":  # averaged over detector and season
            _aatm = np.array(
                [
                    -141.31449999999998,
                    -8.256029999999999,
                    0.6132505,
                    -0.025998975,
                    0.4024275,
                ]
            )
            _batm = np.array(
                [
                    1153.0349999999999,
                    1263.3325,
                    1257.0724999999998,
                    404.85974999999996,
                    1.0,
                ]
            )
            _catm = np.array([967990.75, 668591.75, 636790.0, 814070.75, 21426175.0])
            _thickl = np.array(
                [
                    1011.8521512499999,
                    275.84507575000003,
                    51.0230705,
                    2.983134,
                    0.21927724999999998,
                ]
            )
            _hlay = np.array([0.0, 993750.0, 2081250.0, 4150000.0, 6877500.0])
        elif location == "ANTARES/KM3NeT-ORCA":
            if season == "Summer":
                _aatm = np.array([-158.85, -5.38682, 0.889893, -0.0286665, 0.50035])
                _batm = np.array([1145.62, 1176.79, 1248.92, 415.543, 1.0])
                _catm = np.array([998469.0, 677398.0, 636790.0, 823489.0, 16090500.0])
                _thickl = np.array(
                    [986.951713, 306.4668, 40.546793, 4.288721, 0.277182]
                )
                _hlay = np.array([0, 9.0e5, 22.0e5, 38.0e5, 68.2e5])
            elif season == "Winter":
                _aatm = np.array([-132.16, -2.4787, 0.298031, -0.0220264, 0.348021])
                _batm = np.array([1120.45, 1203.97, 1163.28, 360.027, 1.0])
                _catm = np.array([933697.0, 643957.0, 636790.0, 804486.0, 23109000.0])
                _thickl = np.array(
                    [988.431172, 273.033464, 37.185105, 1.162987, 0.192998]
                )
                _hlay = np.array([0, 9.5e5, 22.0e5, 47.0e5, 68.2e5])
        elif location == "KM3NeT-ARCA":
            if season == "Summer":
                _aatm = np.array([-157.857, -28.7524, 0.790275, -0.0286999, 0.481114])
                _batm = np.array([1190.44, 1171.0, 1344.78, 445.357, 1.0])
                _catm = np.array([1006100.0, 758614.0, 636790.0, 817384.0, 16886800.0])
                _thickl = np.array(
                    [1032.679434, 328.978681, 80.601135, 4.420745, 0.264112]
                )
                _hlay = np.array([0, 9.0e5, 18.0e5, 38.0e5, 68.2e5])
            elif season == "Winter":
                _aatm = np.array([-116.391, 3.5938, 0.474803, -0.0246031, 0.280225])
                _batm = np.array([1155.63, 1501.57, 1271.31, 398.512, 1.0])
                _catm = np.array([933697.0, 594398.0, 636790.0, 810924.0, 29618400.0])
                _thickl = np.array(
                    [1039.346286, 194.901358, 45.759249, 2.060083, 0.142817]
                )
                _hlay = np.array([0, 12.25e5, 21.25e5, 43.0e5, 70.5e5])
        elif location == "SouthPole":
            if season == "December":
                _aatm = np.array([-128.601, -39.5548, 1.13088, -0.00264960, 0.00192534])
                _batm = np.array([1139.99, 1073.82, 1052.96, 492.503, 1.0])
                _catm = np.array([861913.0, 744955.0, 675928.0, 829627.0, 5.8587010e9])
                _thickl = np.array(
                    [1011.398804, 588.128367, 240.955360, 3.964546, 0.000218]
                )
                _hlay = np.array([0.0, 4.0e5, 1.0e6, 4.0e6, 1.0e7])
            elif season == "June":
                _aatm = np.array(
                    [-163.331, -65.3713, 0.402903, -0.000479198, 0.00188667]
                )
                _batm = np.array([1183.70, 1108.06, 1424.02, 207.595, 1.0])
                _catm = np.array([875221.0, 753213.0, 545846.0, 793043.0, 5.9787908e9])
                _thickl = np.array(
                    [1020.370363, 586.143464, 228.374393, 1.338258, 0.000214]
                )
                _hlay = np.array([0.0, 4.0e5, 1.0e6, 4.0e6, 1.0e7])
            else:
                raise Exception(
                    'CorsikaAtmosphere(): Season "'
                    + season
                    + '" not parameterized for location SouthPole.'
                )
        elif location == "PL_SouthPole":
            if season == "January":
                _aatm = np.array([-113.139, -7930635, -54.3888, -0.0, 0.00421033])
                _batm = np.array([1133.10, 1101.20, 1085.00, 1098.00, 1.0])
                _catm = np.array([861730.0, 826340.0, 790950.0, 682800.0, 2.6798156e9])
                _thickl = np.array(
                    [1019.966898, 718.071682, 498.659703, 340.222344, 0.000478]
                )
                _hlay = np.array([0.0, 2.67e5, 5.33e5, 8.0e5, 1.0e7])
            elif season == "August":
                _aatm = np.array([-59.0293, -21.5794, -7.14839, 0.0, 0.000190175])
                _batm = np.array([1079.0, 1071.90, 1182.0, 1647.1, 1.0])
                _catm = np.array([764170.0, 699910.0, 635650.0, 551010.0, 59.329575e9])
                _thickl = np.array(
                    [1019.946057, 391.739652, 138.023515, 43.687992, 0.000022]
                )
                _hlay = np.array([0.0, 6.67e5, 13.33e5, 2.0e6, 1.0e7])
            else:
                raise Exception(
                    'CorsikaAtmosphere(): Season "'
                    + season
                    + '" not parameterized for location SouthPole.'
                )
        else:
            raise Exception(
                "CorsikaAtmosphere:init_parameters(): Location "
                + str(location)
                + " not parameterized."
            )

        self._atm_param = np.array([_aatm, _batm, _catm, _thickl, _hlay])

        self.location, self.season = location, season
        # Clear cached theta value to force spline recalculation
        self.theta_deg = None

    def depth2height(self, x_v):
        """Converts column/vertical depth to height.

        Args:
          x_v (float): column depth :math:`X_v` in g/cm**2

        Returns:
          float: height in cm
        """
        _aatm, _batm, _catm, _thickl, _hlay = self._atm_param

        if x_v >= _thickl[1]:
            height = _catm[0] * np.log(_batm[0] / (x_v - _aatm[0]))
        elif x_v >= _thickl[2]:
            height = _catm[1] * np.log(_batm[1] / (x_v - _aatm[1]))
        elif x_v >= _thickl[3]:
            height = _catm[2] * np.log(_batm[2] / (x_v - _aatm[2]))
        elif x_v >= _thickl[4]:
            height = _catm[3] * np.log(_batm[3] / (x_v - _aatm[3]))
        else:
            height = (_aatm[4] - x_v) * _catm[4]

        return height

    def get_density(self, h_cm):
        """Returns the density of air in g/cm**3.

        Uses the optimized module function :func:`corsika_get_density_jit`.

        Args:
          h_cm (float): height in cm

        Returns:
          float: density :math:`\\rho(h_{cm})` in g/cm**3
        """
        return corsika_get_density(h_cm, *self._atm_param)
        # return corsika_get_density_jit(h_cm, self._atm_param)

    def get_mass_overburden(self, h_cm):
        """Returns the mass overburden in atmosphere in g/cm**2.

        Uses the optimized module function :func:`corsika_get_m_overburden_jit`

        Args:
          h_cm (float): height in cm

        Returns:
          float: column depth :math:`T(h_{cm})` in g/cm**2
        """
        return corsika_get_m_overburden(h_cm, *self._atm_param)
        # return corsika_get_m_overburden_jit(h_cm, self._atm_param)

    def rho_inv(self, X, cos_theta):
        """Returns reciprocal density in cm**3/g using planar approximation.

        This function uses the optimized function :func:`planar_rho_inv_jit`

        Args:
          h_cm (float): height in cm

        Returns:
          float: :math:`\\frac{1}{\\rho}(X,\\cos{\\theta})` cm**3/g
        """
        return planar_rho_inv(X, cos_theta, *self._atm_param)
        # return planar_rho_inv_jit(X, cos_theta, self._atm_param)

    def calc_thickl(self):
        """Calculates thickness layers for :func:`depth2height`

        The analytical inversion of the CORSIKA parameterization
        relies on the knowledge about the depth :math:`X`, where
        trasitions between layers/exponentials occur.

        Example:
          Create a new set of parameters in :func:`init_parameters`
          inserting arbitrary values in the _thikl array::

          $ cor_atm = CorsikaAtmosphere(new_location, new_season)
          $ cor_atm.calc_thickl()

          Replace _thickl values with printout.

        """
        from scipy.integrate import quad

        thickl = []
        for h in self._atm_param[4]:
            thickl.append(
                "{0:4.6f}".format(quad(self.get_density, h, 112.8e5, epsrel=1e-4)[0])
            )
        info(5, "_thickl = np.array([" + ", ".join(thickl) + "])")
        return thickl
class LipariAtmosphere(EarthsAtmosphere):
    def __init__(self):
        EarthsAtmosphere.__init__(self)
        
    def get_density(self, h_cm):
        return rho_lipari(h_cm/1e5)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from Corsika_atm import CorsikaAtmosphere
    plt.figure(figsize=(5, 4))
    plt.title("CORSIKA atmospheres")
    
    cka_obj = CorsikaAtmosphere("USStd", None)
    
    alp = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0, 95.0]
    for a in alp:
        cka_obj.set_theta(a)
        h_vec, X_int = cka_obj.hVx_plot(a)
        plt.plot(X_int, h_vec/1e5, lw=1.5, label=f'{a} degrees')
    plt.xscale('log')
    plt.xlim(10**(-3), 10**7)
    plt.ylim(0.0, 100.0)
    plt.xlabel('Slant Depth $X$ (g/cm$^2$)')
    plt.ylabel('Height (km)')
    plt.legend(title='Theta angles')
    plt.grid(True)
    plt.show()
