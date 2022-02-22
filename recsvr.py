# by Christian Leibold 22.2.22

import numpy as np

class subsig:
    def __init__(self,y,t,a):
        self.y=y;
        self.t=t;
        self.a=a;

def modelfunc(uu,mk):
    return np.sum(uu*mk,axis=1)



## reconstruction of the signal give the loads u
def funcfit(uest,signal,N,myk,par):

    if np.isscalar(N):
        yfinal=np.zeros((uest.shape[0],N))
        narr=list(np.arange(0,N))
    else:
        narr=list(N)
        yfinal=np.zeros((uest.shape[0],len(N)))

    if signal.t.nbytes>0:
        tcorr=signal.t-signal.t[0]
        narr=narr-signal.t[0]
    else:
        tcorr=signal.t


    subtr=(np.ones((len(narr),1))*tcorr).transpose()
    minud=(np.ones((signal.t.shape[0],1))*narr)
    kapp=myk(minud-subtr,*par)

    if uest.nbytes>0:
        yfinal=uest@kapp
    
    if signal.y.nbytes>0:
        b=np.mean(signal.y)-np.mean(yfinal);
    else:
        b=0


    yfinal=yfinal+b;  

    return yfinal


############## incremental rsksvr
inc_dvest=np.zeros((0,1))
inc_Gest=np.zeros((0,0))
inc_Npest=0
inc_ids=np.zeros(0)
inc_iprev=0   
inc_att=np.zeros(0)

def initrecsvr():
    global inc_dvest, inc_Gest, inc_Npest, inc_ids, inc_iprev, inc_att
    inc_dvest=np.zeros((0,1))
    inc_Gest=np.zeros((0,0))
    inc_Npest=0
    inc_ids=np.zeros(0)
    inc_iprev=0   
    inc_att=np.zeros(0)


    
def incrsvr(uest,signal,myk,par,Ncut=300):

    global inc_dvest, inc_Gest, inc_Npest, inc_ids, inc_iprev, inc_att

    # read new data
    y=signal.y
    ismp=signal.t
    inc_att=np.append(inc_att,signal.a)

    P=uest.shape[1]
    Tmem = P+1

    #update time
    inc_ids[0:P]=inc_ids[0:P]+ismp-inc_iprev     

    # initializations
    ndim=y.shape[0]
    uout=np.zeros((ndim,Tmem))
    uout[:,0:P]=uest
    upest=np.zeros((ndim,1))

    #for next time step
    inc_ids=np.append(inc_ids,0)
    Ktmp=inc_att[P]*(inc_att[0:P]*myk(inc_ids[0:P],*par))
    Kp=np.zeros((Ktmp.size,1))
    Kp[:,0]=Ktmp  
        
    inc_dvest[0:P]=inc_Gest[0:P,0:P]@Kp     
    inc_Npest=(inc_att[P]**2)*myk(0,*par)- Kp.transpose()@inc_dvest[0:P]
    
    # compute new uest   
           
    errest=y-modelfunc(uest[:,0:P]*inc_att[0:P],myk(inc_ids[0:P],*par))
    upest = errest.reshape((ndim,1))/inc_Npest   
    uout[:,0:P]=uest[:,0:P]-upest@inc_dvest[0:P].transpose()
    uout[:,P]=upest[:,0]

    
    if inc_Gest.shape[0] < P+1:
        tmp = inc_Gest
        inc_Gest=np.zeros((2*(P+1),2*(P+1)))
        inc_Gest[0:P,0:P]=tmp
        tmp=inc_dvest
        inc_dvest=np.zeros((2*(P+1),1))
        inc_dvest[0:P]=tmp
        
    #update G
    dtmp=inc_dvest[0:P]

    if P< Ncut:
        inc_Gest[0:P,0:P] += np.outer(dtmp,dtmp)/inc_Npest
        inc_Gest[P,0:P] = -dtmp.transpose()/inc_Npest
        inc_Gest[0:P,P] = -inc_dvest[0:P,0]/inc_Npest
        inc_Gest[P,P]   = 1./inc_Npest
    else:
        dvs=inc_dvest[P-Ncut:P];
        dvl=np.zeros((P,1));
        dvl[P-Ncut:P]=dvs;
        inc_Gest[P-Ncut:P,P-Ncut:P] += np.outer(dvs,dvs)/inc_Npest
        inc_Gest[P,0:P] = -dvl.transpose()/inc_Npest
        inc_Gest[0:P,P] = -dvl[:,0]/inc_Npest
        inc_Gest[P,P]   = 1./inc_Npest

  
      
    inc_iprev=ismp


    return uout


## remove pattern p_rem
def decrsvr(uest,p_rem, myk,par,Ncut=300):

    global inc_Gest, inc_ids, inc_att, inc_dvest


    P=uest.shape[1]
    N=uest.shape[0]
    remainder=list(np.append(np.arange(0,p_rem), np.arange(p_rem+1,P)))

    D=inc_Gest[p_rem,p_rem]
    A=np.zeros((P-1,P-1))
    m=0
    for n in remainder:
        A[m,:]=inc_Gest[n,remainder]
        m+=1
        
    B=inc_Gest[remainder,p_rem].reshape((P-1,1))
    A = A-(B@B.transpose())/D

    #######
    dec_ids=-inc_ids[remainder]+inc_ids[p_rem]
    #######
    
    Kp=myk(dec_ids,*par).reshape((P-1,1))
    
    v=uest[:,remainder]
    urem=uest[:,p_rem].reshape((N,1))
    
    #print(v.shape, urem.shape, (A@Kp).transpose().shape)

    inc_dvest=A@Kp
    
    v=v+urem@(inc_dvest.transpose())


    
    inc_att=inc_att[remainder]
    inc_ids=inc_ids[remainder]
    inc_Gest=A

    
    return v
