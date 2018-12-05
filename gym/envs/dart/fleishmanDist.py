import numpy as np
from numpy.linalg import solve
from scipy.stats import moment,norm
from scipy.spatial import ConvexHull, Delaunay

class flDistCalc(object):
    #using algorithm described below, build distribution exhibiting known 4 moments
    #from https://support.sas.com/publishing/authors/extras/65378_Appendix_D_Functions_for_Simulating_Data_by_Using_Fleishmans_Transformation.pdf
    #correcting for coefficient errors (from https://www.diva-portal.org/smash/get/diva2:407995/FULLTEXT01.pdf) and using excess kurtosis explicitly
    
    def __init__(self,_data, showSimRes=True, isData=True):
        if(isData) : #build from data
            self.setDataAndCalc(_data=_data,showSimRes=showSimRes)    
        else :      #build from passed 4 moments
            self.setMmntsAndCalc(mmnts=_data,showSimRes=showSimRes)
 
    #kurtosis is expected to be full kurtosis, not excess
    def setMmntsAndCalc(self, mmnts, isExKurt=False, hasMinMax=True, showSimRes=True, N=1000000):
        self.mmnts = mmnts[:]
        self.mean = mmnts[0]
        self.std = mmnts[1]
        self.skew = mmnts[2]
        self.kurt = mmnts[3]
        #must be excess kurtosis so subtract 3 if isn't
        exkurt = self.kurt-3.0
        if (isExKurt) : 
            exkurt=self.kurt
            self.mmnts[3] +=3.0
            self.kurt += 3.0 
        if hasMinMax :
            self.min=self.mmnts[4] 
            self.max=self.mmnts[5] 
        #coeffs need to be 4 elements pre-pend -coeff[1] to coeffs array
        self._endInitCalc(self.skew,exkurt, showSimRes, N)        

    def setDataAndCalc(self, _data, showSimRes=True, N=1000000):
        #data must be standardized
        self.mean = np.mean(_data)
        self.std = np.std(_data)
        std_data = (_data - self.mean)/self.std
        self.skew = moment(std_data,3)
        self.kurt = moment(std_data,4)
        exkurt = self.kurt - 3.0#need to remove 3 for standard distribution ->excess kurtosis
        self.min=min(_data)
        self.max=max(_data)
        self.mmnts=[self.mean,self.std,self.skew,self.kurt, self.min, self.max]
        self._endInitCalc(self.skew,exkurt, showSimRes, N)
       
    def _endInitCalc(self,skew, exkurt, showSimRes, N):
        self.coeff = self.fit_fleishman_from_sk(skew,exkurt)
        if showSimRes :
            sim = self.genData(N)
            self.dispRes(sim)

    def dispRes(self, sim):
        print('descirptive coeffs : {}\n'.format(self.coeff))
        print('given data moments : {}'.format(self.mmnts))
        print('simulated dist moments : {} '.format(self.moments(sim)))

    #use specified coeffs and data moments to generate fleishman approx of dist with specified 4 moments
    def genData(self, N=10000, clipRes=False):
        if clipRes : #keep results within bounds of min/max while preserving moments
            sim = np.zeros(N)
            firstElem = 0
            lastElem = 0
            while lastElem < N :
                numToGen = N - lastElem
                simTest = (((self.generate_fleishman(*self.coeff,N=numToGen)) * self.std) + self.mean)
                simInBnds = [x for x in simTest if (x < self.max) and (x > self.min)]
                numToAdd = len(simInBnds)
                lastElem = numToAdd + firstElem
                if lastElem > N : 
                    lastElem = N
                sim[firstElem:lastElem] = simInBnds
                firstElem = lastElem    
            #sim=sim[:N]
        else :            
            sim = (((self.generate_fleishman(*self.coeff,N=N)) * self.std) + self.mean)
        return sim

 
    def flfunc(self, b, c, d, skew, kurtosis):        
        #Given the fleishman coefficients, and a target skew and kurtois
        #this function will have a root if the coefficients give the desired skew and kurtosis
        #calculate the variance, skew and kurtois of a Fleishman distribution
        #F = -c + bZ + cZ^2 + dZ^3, where Z ~ N(0,1)

        b2 = b * b
        c2 = c * c
        d2 = d * d
        bd = b * d
        _v = b2 + 6*bd + 2*c2 + 15*d2
        _s = 2 * c * (b2 + 24*bd + 105*d2 + 2)
        _k = 24 * (bd + c2 * (1 + b2 + 28*bd) + 
                    d2 * (12 + 48*bd + 141*c2 + 225*d2))
        return _v - 1, _s - skew, _k - kurtosis

    def flDeriv(self, b, c, d):
        #The deriviative of the flfunc above
        #returns a matrix of partial derivatives
        b2 = b * b
        c2 = c * c
        d2 = d * d
        bd = b * d
        df1db = 2*b + 6*d
        df1dc = 4*c
        df1dd = 6*b + 30*d
        df2db = 4*c * (b + 12*d)
        df2dc = 2 * (b2 + 24*bd + 105*d2 + 2)
        df2dd = 4 * c * (12*b + 105*d)
        df3db = 24 * (d + c2 * (2*b + 28*d) + 48 * d**3)
        df3dc = 48 * c * (1 + b2 + 28*bd + 141*d2)
        df3dd = 24 * (b + 28*b * c2 + 2 * d * (12 + 48*bd + 
                    141*c2 + 225*d2) + d2 * (48*b + 450*d))
        return np.matrix([[df1db, df1dc, df1dd],
                        [df2db, df2dc, df2dd],
                        [df3db, df3dc, df3dd]])

    def newton(self, a, b, c, skew, kurtosis, max_iter=25, lim=1e-5):
        #Implements newtons method to find a root of flfunc
        f = self.flfunc(a, b, c, skew, kurtosis)
        for i in range(max_iter):
            if max(map(abs, f)) < lim:
                break
            J = self.flDeriv(a, b, c)
            delta = -solve(J, f)
            (a, b, c) = delta + (a,b,c)
            f = self.flfunc(a, b, c, skew, kurtosis)
        return (a, b, c)

    #Find an initial estimate of the fleisman coefficients, to feed to newtons method
    def fleishmanic(self, skew, kurt):
        c1 = 0.95357 - 0.05679 * skew + 0.03520 * skew**2 + 0.00133 * kurt**2
        c2 = 0.10007 * skew + 0.00844 * skew**3
        c3 = 0.30978 - 0.31655 * c1
        return (c1, c2, c3)

    #Find the fleishman distribution with given skew and kurtosis
    #mean =0 and stdev =1        
    #Returns list of 0 as coeffs if given kurtosis and skew are infeasible
    def fit_fleishman_from_sk(self, skew, kurt):
        #first verify kurt lies within feasible bound vs skew
        #this is fleish's bound from his paper - said to be wrong in subsequent 2010 paper
        #bound = -1.13168 + 1.58837 * skew**2
        bound = -1.2264489 + 1.6410373*skew**2
        if kurt < bound:
            print('!!!! Coefficient error : kurt {} is not feasible with skew :{}'.format(kurt,skew))
            return np.array([0,0,0,0])
        a, b, c = self.fleishmanic(skew, kurt)
        coeff = self.newton(a, b, c, skew, kurt)
        coeffs = np.array([-coeff[1], *coeff])
        #print('Coeffs : {}'.format(coeffs))
        return coeffs
    
    #Return summary statistics of as set of data 
    def moments(self, data):
        mean = np.mean(data)
        var = np.var(data)
        skew = moment(data,3)/var**1.5
        kurt = moment(data,4)/var**2
        mind = min(data)
        maxd = max(data)
        std = var**.5
        return (mean,std,skew,kurt, mind, maxd)

    def generate_fleishman(self, a,b,c,d,N=100):
        #Generate N data items from fleishman's distribution with given coefficents
        Z = norm.rvs(size=N)
        F = a + Z*(b +Z*(c+ Z*d))
        return F

#a multivariate version of the fleshman transform-based distribution synthesiser
class MV_flDistCalc():
    #build object to simulate nonnormal multivariate data given moments and correlations between variates
    def __init__(self,_data, showSimRes=False, isData=True):
        if isData:#given full data set
            self.setMVDistFromData(_data,showSimRes)
        else :#given moments, correlation matrix, and delauney points
            self.setMVDistFromMmmnts(_data,showSimRes)
    
    #takes multivariate data - list of data points - and builds distribution consistent with covariance of data cols
    def setMVDistFromData(self, _data,showSimRes):
        self.hasHull = False
        #build convex hull and delauney traingularization of convex hull
        #find hull 
        print('{}'.format(_data.shape))
        hull= ConvexHull(_data)    
        #send hull surface points set to build del hull - smaller set, easier to use/save than source points
        self.buildDelHull(hull.points[hull.vertices])

        #_data should be numpy matrix of rows of sample values where we want to build dist based on correlations of data cols
        self.numDims = len(_data[0])
        self.corrMat = np.corrcoef(np.transpose(_data))#should be numDims x numDims
        perDimFlDists = [None]*self.numDims
        perDimCoeffs = [None]*self.numDims
        for i in range(self.numDims):#build dist object for each dimension
            perDimFlDists[i] = flDistCalc(_data[:,i], showSimRes=showSimRes, isData=True)
            perDimCoeffs[i] = perDimFlDists[i].coeff

        self.perDimFlDists = perDimFlDists
        self.perDimCoeffs = perDimCoeffs

    #_mmntsCorDel is dictionary of arrays holding all momments, correlation matrix, and matrix of hull points for delaunay triangularization
    def setMVDistFromMmmnts(self, _mmntsCorDel, showSimRes):
        #build delaunay tri for hull membership calc
        self.buildDelHull(_mmntsCorDel['delHullPts'])
        self.corrMat = _mmntsCorDel['corrMat']
        #matrix of inter-dim correlations
        self.numDims = len(self.corrMat[0])
        perDimFlDists = [None]*self.numDims
        perDimCoeffs = [None]*self.numDims
        #matrix of per-dimension moments and min/max
        mmnts = _mmntsCorDel['mmnts']
        for i in range(self.numDims):#build dist object for each dimension
            perDimFlDists[i] = flDistCalc(mmnts[i,:], showSimRes=showSimRes, isData=False)
            perDimCoeffs[i] = perDimFlDists[i].coeff
        
        self.perDimFlDists = perDimFlDists
        self.perDimCoeffs = perDimCoeffs

    #build delauney triangularization hull of src data points to exclude outliers - uses hull surface points set  
    def buildDelHull(self, hullPts):
        #get delauney triangularization of hull points (much smaller set than size of _data)
        self.delaHull= Delaunay(hullPts)
        self.hasHull=True        

    #multivariate find correlations - finds correlations r between 2 bivariate normal vars
    #Solve the Vale-Maurelli cubic equation to find the intermediate correlation between two normal variables that gives rise to a target
    #correlation (rho) between the two transformed nonnormal variables
    def solveCorr(self, rho, cf1, cf2):
        d1d2 = cf1[3]*cf2[3]
        c1c2 = cf1[2]*cf2[2]
        b1b2 = cf1[1]*cf2[1]
        b1d2 = cf1[1]*cf2[3]
        b2d1 = cf1[3]*cf2[1]
        #cubic inteprolation of coefficients of the 2 rvs
        coefs = np.array([6.0*d1d2, 2.0*c1c2, b1b2 + (3.0*b1d2) + (3.0*b2d1) + (9.0*d1d2), -rho])
        #find roots of polynomial
        roots = np.roots(coefs)
        realRoots = roots[np.isreal(roots)]
        #return smallest real root - since coeffs are all real and odd degree polynomial, will always have at least 1 real root
        root = min(realRoots)
        return root.real

    #call solveCorr for each pair of variables
    # Given a target correlation matrix, tarRmat, and coeffs for each marginal 
    # distribution, find the "intermediate" correlation matrix, V
    def vmTargetCorr(self):
        tarRmat = self.corrMat
        V = np.ones(shape=tarRmat.shape)
        numVars = self.numDims
        cVals = self.perDimCoeffs
        for i in range(1,numVars):
            for j in range(i):
                V[i,j]=self.solveCorr(tarRmat[i,j],cVals[i], cVals[j])
                V[j,i]=V[i,j]
        return V

    #Simulate data from a multivariate nonnormal distribution such that
    #Each marginal distribution has a specified skewness and kurtosis
    #The marginal variables have the correlation matrix Rmat
    def randValeMaurelli(self, N):
        #compute Fleishman coefficients matching marginal moments and intermediate corr mat V
        numVars = self.numDims
        V = self.vmTargetCorr()
        X = norm.rvs(size=(N,numVars))
        cVals = self.perDimCoeffs
        #eigen value decomp here on V -> U * diag(Dvec) * U^-1 == V
        Dvec, U = np.linalg.eig(V)
        #print('Dvec : {}'.format(Dvec))
        #F is sqrt mat for V
        F = np.diag(np.sqrt(Dvec)).dot(np.transpose(U))
        #correlated norms
        Y = X.dot(F)
        for i in range(numVars):
            Z = Y[:,i]
            X[:,i]=((cVals[i][0] + Z*(cVals[i][1] + Z*(cVals[i][2] +  Z*cVals[i][3])))  * self.perDimFlDists[i].std) +  self.perDimFlDists[i].mean
        return X
    
    #restrict results to lie within convex hull if hull is present
    def genMVData(self, N, doReport=False, debug=False, useHull=True):
        if self.hasHull and useHull : 
            #reject samples that are not inside hull of original data
            sim = np.zeros(shape=(N,self.numDims))
            firstElem = 0
            lastElem = 0  
            while lastElem < N :
                numToGen = N - lastElem
                numToAdd = 0
                while numToAdd < 1 :#get some values to add
                    simTest = self.randValeMaurelli(N=numToGen)
                    simInBnds = [x for x in simTest if self.delaHull.find_simplex(x)>=0]
                    numToAdd = len(simInBnds)
                lastElem = numToAdd + firstElem
                if lastElem > N : 
                    lastElem = N       
                if debug :
                    print('num to add {}, first {}, last {}'.format(numToAdd, firstElem,lastElem))
                sim[firstElem:lastElem,:] = simInBnds
                firstElem=lastElem                
                #sim = np.concatenate([sim, np.array(simInBnds)])            
           
        else : #not bounding if no hull
            sim=self.randValeMaurelli(N=N)
        if(doReport):
            self.getReport(sim)
            
        return sim
    
    def getReport(self, sim):
        print('Mmnts of src data (mu,std,skew,kurt, min, max) : ')
        for d in range(len(self.perDimFlDists)) : 
            print('{} : {}'.format(d, self.perDimFlDists[d].mmnts))        
        print('Correlation of src data : \n{}'.format(self.corrMat))

        mmntsAra, corr = self.momentsAndCorr(sim)
        print('Mmnts of simmulated data (mu,std,skew,kurt, min, max) : ')
        for d in range(len(mmntsAra)) : 
            print('{} : {}'.format(d, mmntsAra[d]))        
        print('Correlation of simmed data : \n{}'.format(corr))
        return sim


    def momentsAndCorr(self, dataMat):
        #Return summary statistics of as set of data
        mmntsAra = []
        
        for i in range(len(dataMat[0])):
            data = dataMat[:,i]
            mean = np.mean(data)
            var = np.var(data)
            skew = moment(data,3)/var**1.5
            kurt = moment(data,4)/var**2
            mind = min(data)
            maxd = max(data)
            std=var**.5
            mmntsAra.append([mean,std,skew,kurt, mind, maxd])
        corr = np.corrcoef(np.transpose(dataMat))
        return mmntsAra, corr
