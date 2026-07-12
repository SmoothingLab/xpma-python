from .ema import EMA
from .reverse_ema import ReverseEMA
from .multi_ema import MultiEMA
from .reverse_multi_ema import ReverseMultiEMA
from .ifema import IFEMA
from .eifema import EIFEMA
from .xepma import XEPMA
from .quadratic_xepma import QuadraticXEPMA
from .damped_xepma import DampedXEPMA
from .xpma import XPMA
from .lag_reduction import max_monotone_lag_reduction, lead_ema_max_lag_reduction
from .r_crit import (
    r_crit_m, r_crit_o, r_crit_o_effective, r_crit_c, tau0_m, tau_p_o, tau1_c,
)
from .fractional_smoothness import FractionalSmoothness
from .fast_ema import FastEMA
from .lead_ema import LeadEMA
from .convex_fast_ema import ConvexFastEMA
from .convex_lead_ema import ConvexLeadEMA
from .apex_fast_ema import ApexFastEMA
from .apex_lead_ema import ApexLeadEMA
from .secant_solver import SecantSolver
from .reverse_filter import ReverseFilter
from .lead_xepma import LeadXEPMA
