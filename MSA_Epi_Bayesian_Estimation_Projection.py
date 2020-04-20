import csv
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
import scipy.stats as stats
from scipy.stats import poisson
from scipy.stats import gamma
from scipy.stats import nbinom
import numpy as np
from numpy import log as ln
import statsmodels.api as sm
from datetime import datetime

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def isInt(astring):
    """ Is the given string an integer? """
    try: int(astring)
    except ValueError: return 0
    else: return 1

infperiod=4.5 # infectious period in days


nameMSA="Boston" # write part of the MSA name here, it will find it automatically

# county to MSA conversion
gc2MSA=open('MSA_County_2018.csv', 'r')
readerc2MSA=csv.reader(gc2MSA)

Cname=[]
Cstate=[]
MSAname=[]
MSAFIPS=[]
CFIPS=[]
for row in readerc2MSA:
    if (isInt(row[0])==1):
        #print(row[0],row[3],row[7].replace('County',''),row[8],int(row[9]+row[10]))
        
        Cname.append(row[7].replace('County',''))
        Cstate.append(row[8])
        CFIPS.append(row[9]+row[10])
        MSAname.append(row[3])
        MSAFIPS.append(row[0])


gMSA=open('MSA_Population_from_API.csv', 'r')
readerMSA=csv.reader(gMSA)
uMSAFIPS=[]
MSApop=[]
for row in readerMSA:
    uMSAFIPS.append(row[0])
    MSApop.append(row[1])
#print(uMSAFIPS)

#pMSAFIPS="19100"

FIPSlist=[]
for i in range (len(Cname)):
    #    if (MSAFIPS[i]==pMSAFIPS):
    if (nameMSA in MSAname[i]):
        print(Cname[i],CFIPS[i],MSAname[i])
        FIPSlist.append(CFIPS[i])
        name=MSAname[i].replace(",","")

g=open('us-counties.csv', 'r') # epidemic data for counties
reader=csv.reader(g)
cases=[]
deaths=[]
dates=[]

for row in reader:
    if(row[3] in FIPSlist):
        #print(row[0],row[1],row[2],row[3],row[4],row[5])
        date_object = datetime.strptime(row[0], '%Y-%m-%d').date()
        flag=0
        for i in range(len(dates)):
            if (date_object==dates[i]):
                cases[i]+=int(row[4])
                deaths[i]+=int(row[5])
                flag=1
        if (flag==0):
            dates.append(date_object)
            cases.append(int(row[4]))
            deaths.append(int(row[5]))

#print(dates)
#print(cases)
#print(deaths)

xx=[]
for i in range (len(cases)):
    if (cases[i]>0):
        xx.append(i)
print('days with cases=',len(xx))


##### estimation and prediction
fig, ax = plt.subplots()
xxx=dates[1:]

xx = [x - xx[1] for x in xx]
x=xx[1:]
y=np.diff(cases)


plt.plot(xxx,y,'go',alpha=0.5,markersize=8,label='Reported daily new cases')
sdays=5
yy=smooth(y,sdays) # smoothing over sdays (number of days) moving window, averages large chunking in reporting in consecutive days
yy[-2]=(yy[-4]+yy[-3]+yy[-2])/3. # these 2 last lines should not be necesary but the data tend to be initially underreported and also the smoother struggles.
yy[-1]=(yy[-3]+y[-2]+y[-1])/3.
lowess = sm.nonparametric.lowess # the lowess tracker is very sensitive to frac

z = lowess(y, x)
w = lowess(y, x, frac=1./2.)
lwx = list(zip(*w))[0]
lwy = list(zip(*w))[1]

### plot cases
plt.title(name,fontsize=16)
plt.plot(xxx, yy, 'b-', lw=2,label='Smoothened case time series')
plt.plot(xxx, lwy, 'r-', lw=2,label='Lowess Smoothened case time series')
plt.ylabel("Observed New Cases",fontsize=16)
plt.xlabel("Day",fontsize=20)
plt.tight_layout()
plt.legend()
fig.autofmt_xdate()
str='Daily_New_Cases_w_Smoothing_'+name+'.pdf'
plt.savefig(str)
#plt.show()
plt.clf()

yyy =np.cumsum(yy) # These are confirmed cases after smoothing: tried also a lowess smoother but was a bit more parameer dependent from place to place.
lyyy=np.cumsum(lwy)
TotalCases=yyy

#norm = colors.Normalize(vmin=1, vmax=2*len(TotalCases))
#sm = cm.ScalarMappable(norm, cmap=cm.Paired)
#cnt = 1

alpha=3. # shape parameter of gamma distribution
beta=2.  # rate parameter of gamma distribution see https://en.wikipedia.org/wiki/Gamma_distribution

valpha=[]
vbeta=[]
count=0

pred=[]
pstdM=[]
pstdm=[]
xx=[]
cnt=0
NewCases=[]

predR=[]
pstRRM=[]
pstRRm=[]

anomalyday=[]
anomalypred=[]

for i in range(2,len(TotalCases)):
    
    new_cases=TotalCases[i]-TotalCases[i-1]
    old_new_cases=TotalCases[i-1]-TotalCases[i-2]
    
    # This uses a conjugate prior as a Gamma distribution for b_t, with parameters alpha and beta
    alpha =alpha+new_cases
    beta=beta +old_new_cases
    valpha.append(alpha)
    vbeta.append(beta)
    
    #x = np.linspace(gamma.ppf(0.01, a=alpha, scale=1./beta),gamma.ppf(0.99, a=alpha, scale=1./beta), 100) # this is b
    #xx=1.+infperiod*np.log(x) # this is RR tge reproductive number
    
    #y1 = stats.gamma.pdf(x, a=alpha, scale=1./beta)
    mean, var, skew, kurt = gamma.stats(a=alpha, scale=1/beta, moments='mvsk')
    
    #print(mean,var)
    x=mean
    RRest=1.+infperiod*ln(x)
    if (RRest<0.): RRest=0.
    
    predR.append(RRest)
    testRRM=1.+infperiod*ln( gamma.ppf(0.99, a=alpha, scale=1./beta) )# these are the boundaries of the 99% confidence interval  for new cases
    if (testRRM <0.): testRRM=0.
    pstRRM.append(testRRM)
    testRRm=1.+infperiod*ln( gamma.ppf(0.01, a=alpha, scale=1./beta) )
    if (testRRm <0.): testRRm=0.
    pstRRm.append(testRRm)
    
    
    #print('estimated RR=',RRest,testRRm,testRRM) # to see the numbers for the evolution of Rt
    
    if (new_cases>0. and old_new_cases>0.):
        NewCases.append(new_cases)
        
        # Using a Negative Binomial as the  Posterior Predictor of New Cases, given old one
        # This takes parameters r,p which are functions of new alpha, beta from Gamma
        r, p = alpha, beta/(old_new_cases+beta)
        mean, var, skew, kurt = nbinom.stats(r, p, moments='mvsk')
        
        pred.append(mean) # the expected value of new cases
        testciM=nbinom.ppf(0.99, r, p) # these are the boundaries of the 99% confidence interval  for new cases
        pstdM.append(testciM)
        testcim=nbinom.ppf(0.01, r, p)
        pstdm.append(testcim)
        
        np=p
        nr=r
        flag=0
        while (new_cases>testciM or new_cases<testcim):
            if (flag==0):
                anomalyday.append(dates[i+1]) # the first new cases are at i=2
                anomalypred.append(new_cases)
            
            #print("anomaly",testcim,new_cases,testciM,nr,np) #New  cases when falling outside the 99% CI
            #annealing: increase variance so as to encompass anomalous observation: allow Bayesian code to recover
            # mean of negbinomial=r*(1-p)/p  variance= r (1-p)/p**2
            # preserve mean, increase variance--> np=0.8*p (smaller), r= r (np/p)*( (1.-p)/(1.-np) )
            # test anomaly
            
            nnp=0.95*np # this doubles the variance, which tends to be small after many Bayesian steps
            nr= nr*(nnp/np)*( (1.-np)/(1.-nnp) ) # this assignement preserves the mean of expected cases
            np=nnp
            mean, var, skew, kurt = nbinom.stats(nr, np, moments='mvsk')
            testciM=nbinom.ppf(0.99, nr, np)
            testcim=nbinom.ppf(0.01, nr, np)
            
            flag=1
        else:
            if (flag==1):
                alpha=nr  # this updates the R distribution  with the new parameters that enclose the anomaly
                beta=np/(1.-np)*old_new_cases
                
                testciM=nbinom.ppf(0.99, nr, np)
                testcim=nbinom.ppf(0.01, nr, np)
                
                #pstdM=pstdM[:-1] # remove last element and replace by expanded CI for New Cases
                #pstdm=pstdm[:-1]  # This (commented) in  order to show anomalies, but on
                #pstdM.append(testciM) # in the parameter update, uncomment and it will plot the actual updated CI
                #pstdm.append(testcim)
                
                
                # annealing leaves the RR mean unchanged, but we need to adjus its widened CI:
                testRRM=1.+infperiod*ln( gamma.ppf(0.99, a=alpha, scale=1./beta) )# these are the boundaries of the 99% confidence interval  for new cases
                if (testRRM <0.): testRRM=0.
                testRRm=1.+infperiod*ln( gamma.ppf(0.01, a=alpha, scale=1./beta) )
                if (testRRm <0.): testRRm=0.
                
                pstRRM=pstRRM[:-1] # remove last element and replace by expanded CI for RRest
                pstRRm=pstRRm[:-1]
                pstRRM.append(testRRM)
                pstRRm.append(testRRm)

#print('corrected RR=',RRest,testRRm,testRRM) # to see the numbers for the evolution of Rt

#print("anomaly resolved",i,testcim,new_cases,testciM) # the stats after anomaly resolution



    else:
        NewCases.append(new_cases)
        #pred.append(0.)
        #pstdm.append(0.)
        #pstdM.append(5.)
    
        pred.append(0.)
        pstdM.append(10.)  # not sure this has been included
        pstdm.append(0.)

# visualization of the time evolution of R_t with confidence intervals
plt.clf()
fig, ax = plt.subplots()
#x=[]
#for i in range(len(predR)):
#    x.append(i)
days=dates[3:]
x=days

plt.title(name,fontsize=16)
plt.fill_between(x,pstRRM, pstRRm,color='gray', alpha=0.3,label="99% Confidence Interval")

plt.plot(x,predR,'m-',lw=4,alpha=0.5,ms=8,label=r"Estimated $R_t$")
plt.plot(x,predR,'ro',ms=5,alpha=0.8)
plt.plot(x,pstRRM,'r-',alpha=0.8)
plt.plot(x,pstRRm,'r-',alpha=0.8)
plt.plot( (x[0],x[-1]),(1,1),'k-',lw=4,alpha=0.4)
plt.xlim(x[10],x[-2])
plt.ylim(0,predR[10]+1)
plt.ylabel(r"Estimated $R_t$",fontsize=14)
plt.xlabel("Day",fontsize=20)
plt.tight_layout()
plt.legend()
str='Rt_w_Uncertainty_'+name+'.pdf'
fig.autofmt_xdate()
plt.savefig(str)
#plt.show()



# time series of new cases vs. real time prediction with anomalies
plt.clf()
fig, ax = plt.subplots()

fig.autofmt_xdate()
plt.title(name,fontsize=16)
plt.fill_between(x,pstdM, pstdm,color='gray', alpha=0.3,label="99% Confidence Interval")
plt.plot(x,pred,'c-',lw=4,alpha=0.8,label="Expected Daily Cases")
plt.plot(x,pstdM,'k-',alpha=0.4)
plt.plot(x,pstdm,'k-',alpha=0.4)
plt.plot(x,NewCases,'bo',ms=8,alpha=0.3,label="Observed Daily Cases")
plt.plot(anomalyday,anomalypred,'o',c='red',ms=3,label="Anomalies")
plt.ylabel("Observed vs Predicted New Daily Cases",fontsize=14)
plt.xlabel("Day",fontsize=20)
plt.tight_layout()
plt.legend()
str='Observed_vs_Predicted_Daily_Cases_'+name+'.pdf'
plt.savefig(str)
#plt.show()




