import numpy as np
import recsvr as rs
import matplotlib.pyplot as plt

# example how to use recsvr

def kerfunc(x):
  S=5
  r=1.-(np.abs(x)/S);
  r=(r>0)*r
  
  return r

a=np.array([1,1,1,1,1])
y=np.array([[.4,-1,-2.,-.8,0],[2.,2.5,2.7,2.1,1.5]])
tpnts=np.array([0,5,7,12,15])



uest=np.zeros((2,0))
yfit=np.zeros((2,tpnts.size))
tarr=np.zeros(0)
rs.initrecsvr()

for n in range(tpnts.size):
    tarr=np.append(tarr,tpnts[n])
    signal=rs.subsig(y[:,n],tpnts[n],a[n])
    uout=rs.incrsvr(uest,signal,kerfunc,())
    uest=uout          


sigin=rs.subsig(np.zeros((2,0)),tarr,[])
yfit=rs.funcfit(uest,sigin,16,kerfunc,())

irem=2
uest2=rs.decrsvr(uest,irem,kerfunc,())
idremain=list(np.append(np.arange(0,irem), np.arange(irem+1,tarr.size)))
sigin=rs.subsig(np.zeros((2,0)),tarr[idremain],[])
yfit2=rs.funcfit(uest2,sigin,16,kerfunc,())

plt.plot(np.arange(0,16),yfit[0,:],'-')
plt.plot(np.arange(0,16),yfit[1,:],'-')
plt.plot(tarr,y[0,:], '.')
plt.plot(tarr,y[1,:], '.')
plt.plot(np.arange(0,16),yfit2[0,:],'-')
plt.plot(np.arange(0,16),yfit2[1,:],'-')
plt.show()
