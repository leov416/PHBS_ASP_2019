    # -*- coding: utf-8 -*-
"""
Created on Tue Oct 10

@author: jaehyuk
"""

import numpy as np
import scipy.stats as ss
import scipy.optimize as sopt
from . import normal
from . import bsm
import matplotlib.pyplot as plt

'''
Asymptotic approximation for 0<beta<=1 by Hagan
'''
def bsm_vol(strike, forward, texp, sigma, alpha=0, rho=0, beta=1):
    #这个公式在P13 SABR那里有，这个应该是SABR下用HAGAN进行近似的而已，然后比较一下两者的差别？
    if(texp<=0.0):
        return( 0.0 )

    powFwdStrk = (forward*strike)**((1-beta)/2)
    logFwdStrk = np.log(forward/strike)
    logFwdStrk2 = logFwdStrk**2
    rho2 = rho*rho

    pre1 = powFwdStrk*( 1 + (1-beta)**2/24 * logFwdStrk2*(1 + (1-beta)**2/80 * logFwdStrk2) )
  
    pre2alp0 = (2-3*rho2)*alpha**2/24
    pre2alp1 = alpha*rho*beta/4/powFwdStrk
    pre2alp2 = (1-beta)**2/24/powFwdStrk**2

    pre2 = 1 + texp*( pre2alp0 + sigma*(pre2alp1 + pre2alp2*sigma) )

    zz = powFwdStrk*logFwdStrk*alpha/np.fmax(sigma, 1e-32)  # need to make sure sig > 0
    if isinstance(zz, float):
        zz = np.array([zz])
    yy = np.sqrt(1 + zz*(zz-2*rho))

    xx_zz = np.zeros(zz.size)

    ind = np.where(abs(zz) < 1e-5)
    xx_zz[ind] = 1 + (rho/2)*zz[ind] + (1/2*rho2-1/6)*zz[ind]**2 + 1/8*(5*rho2-3)*rho*zz[ind]**3
    ind = np.where(zz >= 1e-5)
    xx_zz[ind] = np.log( (yy[[ind]] + (zz[ind]-rho))/(1-rho) ) / zz[ind]
    ind = np.where(zz <= -1e-5)
    xx_zz[ind] = np.log( (1+rho)/(yy[ind] - (zz[ind]-rho)) ) / zz[ind]

    bsmvol = sigma*pre2/(pre1*xx_zz) # bsm vol
    return(bsmvol[0] if bsmvol.size==1 else bsmvol)

'''
Asymptotic approximation for beta=0 by Hagan
'''
def norm_vol(strike, forward, texp, sigma, alpha=0, rho=0):
    # forward, spot, sigma may be either scalar or np.array. 
    # texp, alpha, rho, beta should be scholar values
    #这个应该是Hagan正态的近似 Imp Vol吧，令P13页beta=0罢了。

    if(texp<=0.0):
        return( 0.0 )
    
    zeta = (forward - strike)*alpha/np.fmax(sigma, 1e-32)
    # explicitly make np.array even if args are all scalar or list
    if isinstance(zeta, float):
        zeta = np.array([zeta])
        
    yy = np.sqrt(1 + zeta*(zeta - 2*rho))
    chi_zeta = np.zeros(zeta.size)
    
    rho2 = rho*rho
    ind = np.where(abs(zeta) < 1e-5)
    chi_zeta[ind] = 1 + 0.5*rho*zeta[ind] + (0.5*rho2 - 1/6)*zeta[ind]**2 + 1/8*(5*rho2-3)*rho*zeta[ind]**3

    ind = np.where(zeta >= 1e-5)
    chi_zeta[ind] = np.log( (yy[ind] + (zeta[ind] - rho))/(1-rho) ) / zeta[ind]

    ind = np.where(zeta <= -1e-5)
    chi_zeta[ind] = np.log( (1+rho)/(yy[ind] - (zeta[ind] - rho)) ) / zeta[ind]

    nvol = sigma * (1 + (2-3*rho2)/24*alpha**2*texp) / chi_zeta
 
    return(nvol[0] if nvol.size==1 else nvol)

'''
Hagan model class for 0<beta<=1
'''
class ModelHagan:
    alpha, beta, rho = 0.0, 1.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    bsm_model = None
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.beta = beta
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = bsm.Model(texp, sigma, intr=intr, divr=divr)
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        #求bsm_vol的函数，只是开始先确认一下sigma有没有初始化
        sigma = self.sigma if(sigma is None) else sigma
        texp = self.texp if(texp is None) else texp
        forward = spot * np.exp(texp*(self.intr - self.divr))
        return bsm_vol(strike, forward, texp, sigma, alpha=self.alpha, beta=self.beta, rho=self.rho)
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        #用近似的bsm_vol进行定价罢了
        bsm_vol = self.bsm_vol(strike, spot, texp, sigma)
        return self.bsm_model.price(strike, spot, texp, bsm_vol, cp_sign=cp_sign)
    
    def impvol(self, price, strike, spot, texp=None, cp_sign=1, setval=False):
        #这里先用bsm_model求算出给定参数的隐含波动率，再算forward price
        texp = self.texp if(texp is None) else texp
        vol = self.bsm_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        #求解函数，然后给定其他参数，令这个参数与vol应该是零，这样来求解得出sigma，sigma是需要被求解的。
        #这里已知bsm_vol，已知alpha、beta、rho
        iv_func = lambda _sigma: \
            bsm_vol(strike, forward, texp, _sigma, alpha=self.alpha, rho=self.rho) - vol
        sigma = sopt.brentq(iv_func, 0, 10)
        if(setval):
            self.sigma = sigma
        return sigma
    
    def calibrate3(self, price_or_vol3, strike3, spot, texp=None, cp_sign=1, setval=False, is_vol=True):
        '''  
        Given option prices or bsm vols at 3 strikes, compute the sigma, alpha, rho to fit the data
        If prices are given (is_vol=False) convert the prices to vol first.
        Then use multi-dimensional root solving 
        you may use sopt.root
        # https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.root.html#scipy.optimize.root
        '''
        texp = self.texp if(texp is None) else texp
        #impvol(self, price_in, strike, spot, texp=None, cp_sign=1):
        if not is_vol:
            vol3=[self.bsm_model.impvol(price_or_vol3[i],strike3[i],spot,texp,cp_sign) for i in range(len(strike3))]
            print("price to vol",vol3)
        else:
            vol3=price_or_vol3
            print("vol:",vol3)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        #np.exp to make sure they are all >0 and np.tanh to make sure it is -1 TO 1
        iv_func=lambda result:\
            bsm_vol(strike3,forward,texp,np.exp(result[0]),np.exp(result[1]),np.tanh(result[2]))-vol3
        
        #alpha cannot be 0, rho should be <1 for initial values,sigma cannot be 0
        sol=sopt.root(iv_func,[0,0,0])
        sigma,alpha,rho=np.exp(sol.x[0]),np.exp(sol.x[1]),np.tanh(sol.x[2])
        #set value of the model
        
        if setval:
            self.sigma,self.alpha,self.rho=sigma,alpha,rho
        
        return sigma, alpha, rho
        

'''
Hagan model class for beta=0
'''
class ModelNormalHagan:
    alpha, beta, rho = 0.0, 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    normal_model = None
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.beta = 0.0 # not used but put it here
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = normal.Model(texp, sigma, intr=intr, divr=divr)
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        sigma = self.sigma if(sigma is None) else sigma
        texp = self.texp if(texp is None) else texp
        forward = spot * np.exp(texp*(self.intr - self.divr))
        return norm_vol(strike, forward, texp, sigma, alpha=self.alpha, rho=self.rho)
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1):
        n_vol = self.norm_vol(strike, spot, texp, sigma)
        return self.normal_model.price(strike, spot, texp, n_vol, cp_sign=cp_sign)
    
    def impvol(self, price, strike, spot, texp=None, cp_sign=1, setval=False):
        texp = self.texp if(texp is None) else texp
        vol = self.normal_model.impvol(price, strike, spot, texp, cp_sign=cp_sign)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        
        iv_func = lambda _sigma: \
            norm_vol(strike, forward, texp, _sigma, alpha=self.alpha, rho=self.rho) - vol
        sigma = sopt.brentq(iv_func, 0, 50)
        if(setval):
            self.sigma = sigma
        return sigma

    def calibrate3(self, price_or_vol3, strike3, spot, texp=None, cp_sign=1, setval=False, is_vol=True):
        texp = self.texp if(texp is None) else texp
        if not is_vol:
            vol3=[self.normal_model.impvol(price_or_vol3[i],strike3[i],spot,texp,cp_sign) for i in range(3)]
            print("price to vol",vol3)
        else:
            vol3=price_or_vol3
            print("vol:",vol3)
        forward = spot * np.exp(texp*(self.intr - self.divr))
        
        iv_func=lambda result:\
            norm_vol(strike3,forward,texp,np.exp(result[0]),np.exp(result[1]),np.tanh(result[2]))-vol3
        
        
        sol=sopt.root(iv_func,[1,1,1])
        sigma,alpha,rho=np.exp(sol.x[0]),np.exp(sol.x[1]),np.tanh(sol.x[2])
        if setval:
            self.sigma,self.alpha,self.rho=sigma,alpha,rho
    
        return sigma,alpha,rho # sigma, alpha, rho

'''
MC model class for Beta=1
'''
class ModelBsmMC:
    beta = 1.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = bsm.Model(texp, sigma, intr=intr, divr=divr)
        self.steps=100
        self.path_num=1000
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        ''''
        From the price from self.price() compute the implied vol
        this is the opposite of bsm_vol in ModelHagan class
        use bsm_model
        '''
        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        price=self.price(strike,spot,texp,sigma)
        voll=self.bsm_model.impvol(price, strike, spot, texp, cp_sign=1)
        return voll
        
    
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1,seed_=True):
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        if seed_:
            np.random.seed(12345)
        
        #计算出delta tk
        dtk=texp/self.steps
        logS=np.log(spot)
        #generate W1,W2 with a correlation of rho
        W1=np.random.normal(0, 1,(self.path_num,self.steps))
        Z2=np.random.normal(0, 1,(self.path_num,self.steps))
        W2=self.rho*W1+np.sqrt(1-self.rho**2)*Z2 
        
        # update sigmat and get sigmat in different t
        sig_stage=np.exp(self.alpha*np.sqrt(dtk)*W2-0.5*self.alpha**2*dtk)
        sigt=np.cumprod(sig_stage[:,:-1],axis=1)*sigma
        sigt=np.insert(sigt,0,sigma*np.ones(self.path_num),axis=1)
        
        #update price and get logS in different t
        
        S=spot*np.exp(np.cumsum(sigt*W1*np.sqrt(dtk)-0.5*sigt**2*dtk,axis=1))
        #get the last column and plug in and restore to ST
        price=S[:,-1]
        
        price_=np.mean( np.fmax(cp_sign*(price[:,None] - strike), 0),axis=0)
        return price_

        

'''
MC model class for Beta=0
'''

class ModelNormalMC:
    beta = 0.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    normal_model = None
    
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = normal.Model(texp, sigma, intr=intr, divr=divr)
        self.steps=50
        self.path_num=1000
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        '''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model 
        '''
        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        price=self.price(strike,spot,texp,sigma)
        voll=self.normal_model.impvol(price, strike, spot, texp, cp_sign=1)
        
        return voll
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1,seed_=True):
        
        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        '''
        Your MC routine goes here
        Generate paths for vol and price first. Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        if seed_:
            np.random.seed(12345)
        
        #delta tk
        dtk=texp/self.steps
        #generate W1,W2 with a correlation of rho
        W1=np.random.normal(0, 1,(self.path_num,self.steps))
        Z2=np.random.normal(0, 1,(self.path_num,self.steps))
        W2=self.rho*W1+np.sqrt(1-self.rho**2)*Z2 
        
        # update sigmat and get sigmat in different t
        sig_stage=np.exp(self.alpha*np.sqrt(dtk)*W2-0.5*self.alpha**2*dtk)
        sigt=np.cumprod(sig_stage[:,:-1],axis=1)*sigma
        
        sigt=np.insert(sigt,0,sigma*np.ones(self.path_num),axis=1)
        
        #update price and get S in different t
         
        S=spot+np.sum(sigt*W1*np.sqrt(dtk),axis=1)
        
        price=S
        price_=np.mean( np.fmax(cp_sign*(price[:,None] - strike), 0),axis=0)
        return price_

'''
Conditional MC model class for Beta=1
'''
class ModelBsmCondMC:
    beta = 1.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    bsm_model = None
    '''
    You may define more members for MC: time step, etc
    '''
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=1.0, intr=0, divr=0):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.bsm_model = bsm.Model(texp, sigma, intr=intr, divr=divr)
        self.steps=1000
        self.path_num=1000
        
    def bsm_vol(self, strike, spot, texp=None, sigma=None):
        price=self.price(strike,spot,texp,sigma)
        voll=self.bsm_model.impvol(price, strike, spot, texp, cp_sign=1)
        return voll
        
    
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1,seed_=True):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and BSM price.
        Then get prices (vector) for all strikes
        You may fix the random number seed
        '''
        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        #delta tk
        dtk=texp/self.steps
        if seed_:
            np.random.seed(12345)
        
        #generate normal random variable
        
        W1=np.random.normal(0, 1,(self.path_num,self.steps))
        # update sigmat and get sigmat in different t
        sig_stage=np.exp(self.alpha*np.sqrt(dtk)*W1-0.5*self.alpha**2*dtk)
        sigt=np.cumprod(sig_stage[:,:-1],axis=1)*sigma
        sigt=np.insert(sigt,0,sigma*np.ones(self.path_num),axis=1)
        # using integration
        IT=np.sum(sigt**2*dtk,axis=1)
        
        #(self, strike, spot, texp=None, vol=None, cp_sign=1):
        spot_new=spot*np.exp(self.rho/self.alpha*(sigt[:,-1]-sigma)-self.rho**2/2*IT)
        vol_new=np.sqrt((1-self.rho**2)*IT/texp)
        #this is a scalar output,so strike need to be in a loop 
        price_=[self.bsm_model.price(strike[i],spot_new,texp,vol_new,cp_sign) for i in range(len(strike)) ]
                
        return np.mean(price_,axis=1)
        
        
        
        
        
        
'''
Conditional MC model class for Beta=0
'''
class ModelNormalCondMC:
    beta = 0.0   # fixed (not used)
    alpha, rho = 0.0, 0.0
    texp, sigma, intr, divr = None, None, None, None
    normal_model = None
    
    def __init__(self, texp, sigma, alpha=0, rho=0.0, beta=0.0, intr=0, divr=0):
        self.texp = texp
        self.sigma = sigma
        self.alpha = alpha
        self.rho = rho
        self.intr = intr
        self.divr = divr
        self.normal_model = normal.Model(texp, sigma, intr=intr, divr=divr)
        self.steps=1000
        self.path_num=1000
        
    def norm_vol(self, strike, spot, texp=None, sigma=None):
        '''
        From the price from self.price() compute the implied vol
        this is the opposite of normal_vol in ModelNormalHagan class
        use normal_model
        should be same as norm_vol method in ModelNormalMC (just copy & paste)
        '''
        price=self.price(strike,spot,texp,sigma)
        voll=self.normal_model.impvol(price, strike, spot, texp, cp_sign=1)
        return voll
        
    def price(self, strike, spot, texp=None, sigma=None, cp_sign=1,seed_=True):
        '''
        Your MC routine goes here
        Generate paths for vol only. Then compute integrated variance and normal price.
        You may fix the random number seed
        '''
        
        texp = self.texp if(texp is None) else texp
        sigma = self.sigma if(sigma is None) else sigma
        #delta tk
        dtk=texp/self.steps
        if seed_:
            np.random.seed(12345)
        #generate normal random variable
        
        W1=np.random.normal(0, 1,(self.path_num,self.steps))
        # update sigmat and get sigmat in different t
        sig_stage=np.exp(self.alpha*np.sqrt(dtk)*W1-0.5*self.alpha**2*dtk)
        sigt=np.cumprod(sig_stage[:,:-1],axis=1)*sigma
        sigt=np.insert(sigt,0,sigma*np.ones(self.path_num),axis=1)
        # using integration
        IT=np.sum(sigt**2*dtk,axis=1)
        
        #(self, strike, spot, texp=None, vol=None, cp_sign=1):
        spot_new=spot+self.rho/self.alpha*(sigt[:,-1]-sigma)
        vol_new=np.sqrt((1-self.rho**2)*IT/texp)
        #this is a scalar output,so strike need to be in a loop 
        #price(self, strike, spot, texp=None, vol=None, cp_sign=1):
        price_=[self.normal_model.price(strike[i],spot_new,texp,vol_new,cp_sign) for i in range(len(strike)) ]
                
        return np.mean(price_,axis=1)