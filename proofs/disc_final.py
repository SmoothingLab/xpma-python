import os
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v]="1"
import mpmath as mp
mp.mp.dps=50
# The two classical bounds exactly as written in the proof, checked well beyond (0,ln2):
# (A) sinh(z)/z <= e^{z^2/6}  ; (B) z*coth(z) >= 1
okA=okB=True
for k in range(1,200000):
    z = mp.mpf(k)/20000     # z up to 10
    if mp.sinh(z)/z > mp.e**(z*z/6)*(1+mp.mpf(10)**-40): okA=False
    if z*mp.coth(z) < 1 - mp.mpf(10)**-40: okB=False
print("(A) sinh(z)/z <= e^{z^2/6} for z in (0,10]:", okA)
print("(B) z coth(z) >= 1        for z in (0,10]:", okB)
# and the final assembled bound I(x) >= x(18-x)/24 on the actual s=1 domain
def phi(x): return mp.mpf(1) if x==0 else x/(mp.e**x-1)
ok=True; worst=None
ln2=mp.log(2)
for k in range(1,100000):
    x=ln2*mp.mpf(k)/100000
    I=mp.log(phi(x))+mp.mpf(3)/2*x+phi(x)/2-mp.mpf(1)/2
    lb=x*(18-x)/24
    if I<lb-mp.mpf(10)**-40: ok=False
    if I<=0: ok=False
    if worst is None or I<worst: worst=I
print("I(x) >= x(18-x)/24 and I(x)>0 on (0,ln2):", ok, " min I:", mp.nstr(worst,6))
