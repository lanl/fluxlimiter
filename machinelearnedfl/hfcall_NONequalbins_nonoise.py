import os
import os.path
import sys
import numpy as np
import scipy.io as sio
import dill
from numpy.linalg import inv
from numba import njit, prange
from utnj_dist_anal import limiterJacobianPWNONequalbin as PW_limiter
from utnj_dist_anal import deltarfuncNONequalbin as delta_R
from utnj_dist_anal import phiPWNONequalbin as PW_phi
from utnj_dist_anal import definebin

@njit(target='cpu')
def idxL(i, N, j=1):
    """produce an index array of length 5

    i (int): the central value in the index
    N (int): modulus
    j (int): step

    returns index = [i-2j, i-j, i, i+j, i+2j % N, unless:
      * i < j, then index = [i-2j-1, i-j-1, i, i+j, i+2j] % N
      * j <= i < 2j, then index = [i-2j-1, i-j, i, i+j, i+2j] % N
      * i >= N-j, then index = [i-2j, i-j, i, i+j+1, i+2j+1] % N
      * N-j > i >= N-2j, then index = [i-2j, i-j, i, i+j, i+2j+1] % N
    """
    if i < j:
        return np.array([(i-j*2-1)%N, (i-j-1)%N, i%N, (i+j)%N, (i+j*2)%N])
    if i < j*2:
        return np.array([(i-j*2-1)%N, (i-j)%N, i%N, (i+j)%N, (i+j*2)%N])
    if i >= N-j:
        return np.array([(i-j*2)%N, (i-j)%N, i%N, (i+j+1)%N, (i+j*2+1)%N])
    if i >= N-j*2:
        return np.array([(i-j*2)%N, (i-j)%N, i%N, (i+j)%N, (i+j*2+1)%N])
    return np.array([(i-j*2)%N, (i-j)%N, i%N, (i+j)%N, (i+j*2)%N])


@njit(target='cpu')
def _gendata(t,N,CG,dt,D,G,icg,dat,u):
    for ix in range(0, N):
        ixL = idxL(ix, N)
        ixL0 = np.array([idxL(ixL[1] - 1, N)[0]])
        ixtot = np.hstack((ixL0, ixL))
        ixLcg = idxL(ix, N, j=CG)
        ixLcg0 = np.array([idxL(ixLcg[1] - 1, N, j=CG)[0]])
        ixtotcg = np.hstack((ixLcg0, ixLcg))

        usv = u[ixtot, t-1].copy()
        if t > 1:
            usvcg = u[ixtotcg, t-2].copy()

        L2 = ixL[2]
        L14 = ixL[1:4]
        u[L2, t] = u[L2, t-1] + dt * (
            np.dot(D, u[L14, t-1]) + np.dot(G, u[L14, t-1]**2))

        if t > 1:
            dat[:, icg] = np.hstack((np.array([u[ixLcg[2], t]]), usvcg))
            icg += 1

    return icg,dat,u


@njit(target='cpu')
def _gensim(T,N,CG,dt,D,G,Nrange):
    # generate one simulation
    icg = 0
    _dat = np.zeros((7, Nrange))
    u = np.zeros((N, T));
    u[:, 0] = 2*np.random.rand(N) - 1  #XXX: use random_state?
    for t in range(1, T):
        icg,_dat,u = _gendata(t,N,CG,dt,D,G,icg,_dat,u)
    return _dat


@njit(target='cpu')
def generate_data(T, dt, Ns, nu, X, dx, CG=1):
    """generate dataset for testing and training

    T (float): total time in a simulation
    dt (float): change in time per step
    Ns (int): number of simulations
    nu (float): viscosity
    X (float): total length of a simulation
    dx (float): change in position per step
    CG (int): coarse graining factor

    returns dataset with shape (7, Ns * Nrange), where Nrange = (X/dx+1)*(T-2)
    """ #TODO: state equations used; what is in each of 7 rows
    #T = 800; dt = 0.0005; Ns = 500; nu = 0.01; X = 2; dx = 0.005

    N = int(X/(dx/CG)+1) #NOTE: 1/CG
    Nrange = int(N * (T - 2))
    D = nu/(dx**2) * np.array([1, -2, 1])
    G = 1/(4 * dx) * np.array([1, 0, -1])

    # generate the data
    #print('generating data...')
    dat = np.zeros((7, Ns * Nrange)) # better, empty? #TODO: use 3D array

    for j in prange(0, Ns): # number of simulations
        # generate and store one simulation
        dat[:, j*Nrange:(j+1)*Nrange] = _gensim(T,N,CG,dt,D,G,Nrange)
    return dat


def delta_x(x=2.0, n=201):
    "find change in x for an array of length x with n elements"
    return np.diff(np.linspace(0,x,n))[0]

def vanleer_bb(deltarend):
    "calculate BB given deltaRend for van leer"
    return np.divide(np.diff(np.divide(2.0*deltarend, (deltarend + 1.0))),
                     np.diff(deltarend))


class Limiter(object):

    def __init__(self, pwlimiter, deltaR, phipw, **kwds):

        self._limiter = pwlimiter
        self._deltaR = deltaR
        self._phi = phipw
        self.dt = kwds.get('dt', 1e-3) # change in t (float)
        self.dx = kwds.get('dx', 1e-2) # change in x (float)
        self.nu = kwds.get('nu', 1e-2) # viscosity (float)
        self.jmax = kwds.get('jmax', 20)
        self.phi0 = kwds.get('phi0', 0)
        self.Nitr = kwds.get('Nitr', 1)
        self.Ncal = kwds.get('Ncal', 1) # sims
        self.Nbatch = kwds.get('Nbatch', 1)
        # initialize the following as arrays with Ncal, etc?
        self.deltarend = None
        self.deltarbin = None
        self.Amat = None # for training
        self.Cmat = None # for training
        self.Barr = None # for training
        self.rP = None # for training
        self.rM = None # for training
        self.DF = None # for training
        self._rP = None # for testing
        self._rM = None # for testing
        self._DF = None # for testing

    def Rbins(self, rP=None): #NOTE: core is njit'ed
        if rP is None:
            rP = self.rP
        self.rP = rP
        positive = rP.T[rP.T > 0] #XXX: move inside definebin?
        positive.sort() #XXX: move inside definebin?
        drend, drbin, jmax = definebin(self.jmax, positive)
        self.jmax = jmax
        self.deltarend = np.asarray(drend)
        self.deltarbin = np.asarray(drbin)
        return self.deltarend, self.deltarbin

    def _deltaRi(self, i): #NOTE: core is njit'ed
        rPi = self.rP[i]
        rMi = self.rM[i]
        drPM = np.zeros((self.jmax,2))
        drPM[:,0] = self._deltaR(rPi, self.jmax, self.deltarend, self.deltarbin)
        drPM[:,1] = self._deltaR(rMi, self.jmax, self.deltarend, self.deltarbin)
        return drPM

    def _run(self, uinput, uinput_): #NOTE: core is njit'ed
        "generate rP[i], rM[i], DeltaF_*[i] where *=(1,2,3)"
        dt, dx, nu = self.dt, self.dx, self.nu
        d1, d2, d3, rP, rM = self._limiter(dt, dx, nu, uinput, uinput_)
        if self.deltarend is None:
            return rP, rM
        return rP, rM, d1, d2, d3

    def _err(self, Barr, ri): #NOTE: core is njit'ed
        "call phi, given r*i, where *=(P,M)"
        drend = self.deltarend
        drbin = self.deltarbin
        jmax = self.jmax
        phi0 = self.phi0
        return self._phi(Barr, ri, drend, drbin, jmax, phi0)

    def _delta(self, Barr, rPi, rMi, DFi): #NOTE: core is njit'ed
        return (DFi[0] + self._err(Barr, rPi) * DFi[1]
                       + self._err(Barr, rMi) * DFi[2])

    def _error(self, gt, xdat, Barr, rPi, rMi, DFi): #NOTE: core is njit'ed
        #if Barr is None: Barr = self.Barr
        dt = self.dt
        dx = self.dx
        return .5 * (xdat[3] - dt/dx * self._delta(Barr, rPi, rMi, DFi) - gt)**2.

    def phi(self, data, **kwds):
        "calculate error for Barr for the flux limiter"
        Nitr = kwds.get('Nitr', self.Nitr)
        Ncal = kwds.get('Ncal', self.Ncal)
        nbatch = kwds.get('Nbatch', self.Nbatch)
        compare = kwds.get('compare', None)
        Nprint = kwds.get('Nprint', 0)
        N0 = kwds.get('shift', 0)
        err = np.zeros(Ncal)
        if compare is not None:
            err2 = np.zeros(Ncal)

        if self.Barr is None or self.deltarend is None:
            msg = 'Barr is None. Training has not been performed'
            raise ValueError(msg)
        Barr = self.Barr

        dx = self.dx
        dt = self.dt
        for i_loop in range(Nitr): #NOTE: Nitr == 1
            for it in range(1,Ncal): #XXX: njit '_error'? prange?
                rPi, rMi, DFi = self.rP[it], self.rM[it], self.DF[it]
                batch0 = (N0 + it-1)*nbatch
                for ibatch in range(nbatch): #NOTE: nbatch == 1
                    gt = data[0, batch0 + ibatch]
                    xdat = data[1:7, batch0 + ibatch]
                    err[it] = self._error(gt, xdat, Barr, rPi, rMi, DFi)
                    if compare is not None:
                        err2[it] = self._error(gt, xdat, compare, rPi, rMi, DFi)

                    if Nprint != 0 and it%Nprint == 0 and it/Nprint >= 0:
                        print('i: %s' % it)
                        if compare is None:
                            print('err: %s' % err[it])
                        else:
                            print('err: %s, %s' % (err[it], err2[it]))
        return err if compare is None else (err, err2)

    def test(self, data, **kwds):
        "generate (rP, rM, DeltaF) and error for Barr for the flux limiter"
        Nitr = kwds.get('Nitr', self.Nitr)
        Ncal = kwds.get('Ncal', self.Ncal)
        nbatch = kwds.get('Nbatch', self.Nbatch)
        compare = kwds.get('compare', None)
        Nprint = kwds.get('Nprint', 0)
        N0 = kwds.get('shift', 0)
        run = lambda xdat: self._run(xdat[1:6].copy(), xdat[0:5].copy())
        err = np.zeros(Ncal)
        if compare is not None:
            err2 = np.zeros(Ncal)

        if self.Barr is None or self.deltarend is None:
            msg = 'Barr is None. Training has not been performed'
            raise ValueError(msg)
        Barr = self.Barr

        self._rP = rP = np.zeros(Ncal)
        self._rM = rM = np.zeros(Ncal)
        self._DF = DF = np.zeros((Ncal,3))

        dx = self.dx
        dt = self.dt
        for i_loop in range(Nitr): #NOTE: Nitr == 1
            for it in range(1,Ncal): #XXX: njit 'run', '_error'? prange?
                batch0 = (N0 + it-1)*nbatch
                for ibatch in range(nbatch): #NOTE: nbatch == 1
                    gt = data[0, batch0 + ibatch]
                    xdat = data[1:7, batch0 + ibatch]
                    rP[it], rM[it], DF[it,0], DF[it,1], DF[it,2] = run(xdat)
                    rPi, rMi, DFi = rP[it], rM[it], DF[it]
                    err[it] = self._error(gt, xdat, Barr, rPi, rMi, DFi)
                    if compare is not None:
                        err2[it] = self._error(gt, xdat, compare, rPi, rMi, DFi)

                    if Nprint != 0 and it%Nprint == 0 and it/Nprint >= 0:
                        print('i: %s' % it)
                        print('DF: %s, %s, %s' % (DFi[0], DFi[1], DFi[2]))
                        print('rPM: %s, %s' % (rPi, rMi))
                        if compare is None:
                            print('err: %s' % err[it])
                        else:
                            print('err: %s, %s' % (err[it], err2[it]))

        return err if compare is None else (err, err2)

    def fit(self, data, **kwds): #NOTE: similar to sklearn
        self.__call__(data, **kwds)
        return self

    def error(self, data, **kwds): #NOTE: similar to sklearn
        return self.phi(data, **kwds) #NOTE: but only for training

    def predict_error(self, data, **kwds): #NOTE: similar to sklearn
        return self.test(data, **kwds) #NOTE: but only for testing

    def train(self, data, **kwds):
        self.__call__(data, **kwds)
        return self.phi(data, **kwds)

    def __call__(self, data, **kwds):
        "generate (rP, rM, DeltaF) and (Cmat, Amat) with flux limiter"
        self.Nitr = Nitr = kwds.get('Nitr', self.Nitr)
        self.Ncal = Ncal = kwds.get('Ncal', self.Ncal)
        self.Nbatch = nbatch = kwds.get('Nbatch', self.Nbatch)
        Nprint = kwds.get('Nprint', 0)
        N0 = kwds.get('shift', 0)
        self.rP = np.zeros(Ncal)
        self.rM = np.zeros(Ncal)
        run = lambda xdat: self._run(xdat[1:6].copy(), xdat[0:5].copy())

        if self.deltarend is None:
            for i_loop in range(Nitr): #NOTE: Nitr == 1
                for it in range(1,Ncal): #XXX: njit 'run'? prange?
                    batch0 = (N0 + it-1)*nbatch
                    for ibatch in range(nbatch): #NOTE: nbatch == 1
                        xdat = data[1:7, batch0 + ibatch]
                        self.rP[it], self.rM[it] = run(xdat)

            self.Rbins(self.rP)
            return self.rP, self.rM, None

        # otherwise, we also want DF, Cmat, Amat
        self.DF = np.zeros((Ncal,3))
        drF = np.zeros((self.jmax,Ncal))
        self.Cmat = np.zeros((self.jmax,1))

        dx = self.dx
        dt = self.dt
        for i_loop in range(Nitr): #NOTE: Nitr == 1
            for it in range(1,Ncal): #XXX: njit 'run', '_deltaRi'? prange?
                batch0 = (N0 + it-1)*nbatch
                for ibatch in range(nbatch): #NOTE: nbatch == 1
                    #gt = data[0, batch0 + ibatch]
                    xdat = data[1:7, batch0 + ibatch]
                    self.rP[it], self.rM[it], self.DF[it,0], self.DF[it,1], self.DF[it,2] = run(xdat)

                    if Nprint != 0 and it%Nprint == 0 and it/Nprint >= 0:
                        print('i: %s' % it)
                        print('DF: %s, %s, %s' % (self.DF[it,0], self.DF[it,1], self.DF[it,2]))
                        print('rPM: %s, %s' % (self.rP[it],self.rM[it]))

                    OG = (xdat[3] - dt/dx * self.DF[it,0]
                          - data[0, batch0 + ibatch] # gt
                         ).reshape(-1,1)
                    # product of deltar+- (jmax,2) and deltaF (2,)
                    # jmax x N matrix, each colum is one vector deltarF     
                    drF[:,it] = np.dot(self._deltaRi(it),self.DF[it,1:])
                    self.Cmat += OG * drF[:,it].reshape(-1,1) # (jmax,1)

        self.Cmat *= dx/dt
        self.Amat = np.dot(drF,drF.T).T
        self.Barr = np.dot(inv(self.Amat),self.Cmat)
        return self.rP, self.rM, self.DF


@njit(target='cpu')
def _errmean(i, errT, rPT, deltarend):
    return errT[(rPT > deltarend[i]) & (rPT <= deltarend[i+1])].mean()

@njit(target='cpu')
def _errmeanVL(i, errVLT, rPT, deltarend):
    return errVLT[(rPT > deltarend[i]) & (rPT <= deltarend[i+1])].mean()

#FIXME: better to savemat to a tmpfile, then rename after done writing
def analysis(hyperparam, **kwds):
    nu, CG, jmax, phi0 = hyperparam

    # ensure int hyperparams are ints
    CG = int(CG); jmax = int(jmax)

    import uuid
    version = kwds.get('version', uuid.uuid4().hex[:7])
    shuffle = kwds.get('shuffle', True)
    infiles = kwds.get('infiles', True) # reuse already generated data files
    compare = kwds.get('compare', False) # compare to van leer
    outfiles = kwds.get('outfiles', False) # save some results to mat files
    verbose = kwds.get('verbose', None)
    lock = kwds.get('lock', None)

    import datetime
    start = datetime.datetime.now()
    if verbose is not False: print(start)

    # dataset parameters
    T = kwds.get('T', 800)
    X = 2.0
    dt = 5e-4 * CG # change in t (2*CG)
    dx = 5e-3 * CG # delta_x(x=X, n=401) # change in x

    # learning parameters
    Ns = kwds.get('Ns', 120)
    Nstraining = int(Ns * 5/6.) #XXX: change the ratio?
    nbatch = 1
    Nitr = 1
    Nprint = 500000 if verbose is True else 0

    # prepare learning variables
    N = int(X/(dx/CG)+1)  #NOTE: fixed (at 401) for legacy data
    Nrange = int(N * (T - 2)) #NOTE: should be able to extract from saved data
    Nstest = Ns - Nstraining
    Ncal = int(Nrange * Nstraining / nbatch)

    # 6-point training data with random initial conditions
    mnu = int(nu * 1000)
    inputfile = "6pts_%scg_%sNs_T%s_%smnu_%s.mat" % (int(CG), Ns, T, mnu, version)

    # generate a parallel-aware random state
    from mystic.tools import random_state
    rng = random_state('numpy', seed='*') 

    # get the data
    import hdf5storage
    if os.path.isfile(inputfile) and infiles:
        if verbose is not False: print('file loading...')
        dat = hdf5storage.loadmat(inputfile)['datcg']
        if dat.shape != (7, Ns * Nrange):
            # we have the wrong shape data
            if verbose is not False:
                print('Requested (7, %s) but found %s' % (Ns*Nrange, dat.shape))
                print("Generating train/test data...")
            dat = generate_data(T, dt, Ns, nu, X, dx, CG)
            if verbose is not False: print('saving data...')
            if lock is not None: lock.acquire()
            rnd = rng.randint(1e7)
            fl = os.path.splitext(inputfile)[0]
            hdf5storage.savemat(fl+'_%s.mat' % rnd, {'datcg': dat})
            os.renames(fl+'_%s.mat' % rnd, inputfile)
            if lock is not None: lock.release()
        elif shuffle:
            data = dat.reshape(7, -1, Nrange) #TODO: use in gt,xdat
            rng.shuffle(np.rot90(data))
            dat = dat[:, :(Ns*Nrange)] #NOTE: only using Ns simulations
    else:
        if verbose is not False: print("Generating train/test data...")
        dat = generate_data(T, dt, Ns, nu, X, dx, CG)
        if verbose is not False: print('saving data...')
        if lock is not None: lock.acquire()
        rnd = rng.randint(1e7)
        fl = os.path.splitext(inputfile)[0]
        hdf5storage.savemat(fl+'_%s.mat' % rnd, {'datcg': dat})
        os.renames(fl+'_%s.mat' % rnd, inputfile)
        if lock is not None: lock.release()

    # build the limiter #XXX: build a separate VL Limiter instance?
    #TODO: only save attributes necessary for "predict" and "score"
    mphi = int(phi0 * 1000)
    rootdir = 'run'+str(jmax)+'_%s_%sxCG_%s_%s' % (mphi, int(CG), mnu, version)
    if verbose is not False: print('building limiter...')
    lim = Limiter(PW_limiter, delta_R, PW_phi, dt=dt,
                  dx=dx, nu=nu, jmax=jmax, phi0=phi0,
                  Nitr=Nitr, Ncal=Ncal, Nbatch=nbatch)

    # get rP
    if verbose is not False: print('getting rP...')
    rPfile = rootdir+'/rPvalue_%sxCG_%s.mat' % (CG, mnu)
    rMfile = rootdir+'/rMvalue_%sxCG_%s.mat' % (CG, mnu)
    lmfile = rootdir+'/limiter_'+str(jmax)+'_%s_%sxCG_%s.pkl' % (mphi, CG, mnu)
    if os.path.isfile(lmfile) and infiles:
        with open(lmfile, 'rb') as f:
            lim_ = dill.load(f)
        lim.rP = lim_.rP
        lim.rM = lim_.rM
        del lim_
        lim.Rbins(lim.rP)
    elif os.path.isfile(rPfile) and infiles:
        lim.rP = sio.loadmat(rPfile)['rP']
        lim.rM = sio.loadmat(rMfile)['rM']
        lim.Rbins(lim.rP)
    else: # generate rootdir and rPvalue_NxCG.mat
        lim(dat)
        if not os.path.isdir(rootdir):
            os.mkdir(rootdir)
        #NOTE: duplicate -- saved within limiter.pkl
        #sio.savemat(rPfile, {'rP':lim.rP})
        #sio.savemat(rMfile, {'rM':lim.rM})

    # get deltaR and BB arrays
    if verbose is not False:
        print('deltaR and BB')
        print('deltarend:\n%s' % lim.deltarend.tolist())
        print('deltarbin:\n%s' % lim.deltarbin.tolist())
    if compare is True:
        bbVL = vanleer_bb(lim.deltarend)
        if verbose is not False: print('BBVL:\n%s' % bbVL.tolist())
    else:
        bbVL = None

    # TRAIN and evaluate errors on training points
    if verbose is not False: print('training...')
    if compare is True:
        err, errVL = lim.train(dat, compare=bbVL, Nprint=Nprint)
    else:
        err = lim.train(dat, compare=bbVL, Nprint=Nprint)

    if verbose is not False: print('processing results...')
    #NOTE: duplicate -- saved within limiter.pkl
    #if not os.path.isfile(rPfile):
    #    sio.savemat(rPfile, {'rP':lim.rP})
    #    sio.savemat(rMfile, {'rM':lim.rM})
    #NOTE: duplicate -- saved within limiter.pkl
    #Cfile = rootdir+'/Cmat.mat'
    #Afile = rootdir+'/Amat.mat'
    #bfile = rootdir+'/barray_value.mat'
    #sio.savemat(Cfile, {'C':lim.Cmat})
    #sio.savemat(Afile, {'A':lim.Amat})
    #sio.savemat(bfile, {'b':lim.Barr})
    if outfiles is True:
        if lock is not None: lock.acquire()
        rnd = rng.randint(1e7)
        ol = rootdir+'/error_training_%sxCG_%s_%s.mat' % (CG, mnu, version)
        fl = os.path.splitext(ol)[0]
        sio.savemat(fl+'_%s.mat' % rnd, {'err_tr':err})
        os.renames(fl+'_%s.mat' % rnd, ol)
        if lock is not None: lock.release()
        if compare is True:
            if lock is not None: lock.acquire()
            rnd = rng.randint(1e7)
            ol = rootdir+'/error_trainingVL_%sxCG_%s_%s.mat' % (CG, mnu, version)
            fl = os.path.splitext(ol)[0]
            sio.savemat(fl+'_%s.mat' % rnd, {'err_tr':errVL})
            os.renames(fl+'_%s.mat' % rnd, ol)
            if lock is not None: lock.release()
    if verbose is not False: print('BB.T:\n%s' % lim.Barr.squeeze().tolist())

    # TEST by evaluating errors on test points
    if Nstest > 0:
        if verbose is not False: print('testing...')
        Ncal_test = int(Nrange * Nstest / nbatch) #remaining as holdout data to test
        if compare is True:
            err_test, errVL_test = lim.test(dat, Nitr=1, Ncal=Ncal_test, Nbatch=nbatch, shift=Ncal, compare=bbVL, Nprint=Nprint)
        else:
            err_test = lim.test(dat, Nitr=1, Ncal=Ncal_test, Nbatch=nbatch, shift=Ncal, compare=bbVL, Nprint=Nprint)

        #NOTE: duplicate -- saved within limiter.pkl
        #sio.savemat(rootdir+'/rPvalue_testset_%sxCG_%s.mat' % (CG, mnu),{'rP':lim._rP})
        #sio.savemat(rootdir+'/rMvalue_testset_%sxCG_%s.mat' % (CG, mnu),{'rM':lim._rM})
        if outfiles is True:
            if lock is not None: lock.acquire()
            rnd = rng.randint(1e7)
            ol = rootdir+'/error_testset_%sxCG_%s_%s.mat' % (CG, mnu, version)
            fl = os.path.splitext(ol)[0]
            sio.savemat(fl+'_%s.mat' % rnd, {'err_tr':err_test})
            os.renames(fl+'_%s.mat' % rnd, ol)
            if lock is not None: lock.release()
            if compare is True:
                if lock is not None: lock.acquire()
                rnd = rng.randint(1e7)
                ol = rootdir+'/error_testsetVL_%sxCG_%s_%s.mat' % (CG, mnu, version)
                fl = os.path.splitext(ol)[0]
                sio.savemat(fl+'_%s.mat' % rnd, {'err_tr':errVL_test})
                os.renames(fl+'_%s.mat' % rnd, ol)
                if lock is not None: lock.release()
    ended = datetime.datetime.now()
    if verbose is not False:
        print(ended)
        print(ended - start)

    #TODO: usefiles = True # False for dill
    # save limiter instance #XXX: should use HDF5
    if verbose is not False:
        _phi = int(lim.phi0 * 1000)
        _nu = int(lim.nu * 1000)
        print('{0} == jmax:{1} phi0:{2} CG:{3} nu:{4}'.format(lmfile, lim.jmax, _phi, CG, _nu))
        del _phi, _nu
    with open(lmfile, 'wb') as f:
        dill.dump(lim, f)
        #dill.dump(err, f)
        #if compare is True: dill.dump(errVL, f)
        #dill.dump(err_test, f)
        #if compare is True: dill.dump(errVL_test, f)
    if verbose is not False: print('saved limiter instance')

    #TODO: ANALYZE (should be internal to limiter? or at least a function)
    deltarend = lim.deltarend
    jmax = lim.jmax #NOTE: is len(deltarend) - 1

    # training data
    errT = err.T
    if compare is True: errVLT = errVL.T
    rPT = lim.rP.T

    errmean = np.zeros(jmax)
    if compare is True: errmeanVL = np.zeros(jmax)
    for i in range(0,jmax):
        errmean[i] = _errmean(i, errT, rPT, deltarend)
        if compare is True: errmeanVL[i] = _errmeanVL(i, errVLT, rPT, deltarend)

    if verbose is not False:
        print('for training data')
        print('meanerr: %s' % errmean.tolist())
    if outfiles is True:
        if lock is not None: lock.acquire()
        rnd = rng.randint(1e7)
        ol = rootdir+'/error_mean_bin_training_%sxCG_%s_%s.mat' % (CG, mnu, version)
        fl = os.path.splitext(ol)[0]
        sio.savemat(fl+'_%s.mat' % rnd, {'errmean':errmean})
        os.renames(fl+'_%s.mat' % rnd, ol)
        if lock is not None: lock.release()
    if compare is True:
        if verbose is not False: print('meanerrVL: %s' % errmeanVL.tolist())
        if outfiles is True:
            if lock is not None: lock.acquire()
            rnd = rng.randint(1e7)
            ol = rootdir+'/error_meanVL_bin_training_%sxCG_%s_%s.mat' % (CG, mnu, version)
            fl = os.path.splitext(ol)[0]
            sio.savemat(fl+'_%s.mat' % rnd, {'errmean':errmeanVL})
            os.renames(fl+'_%s.mat' % rnd, ol)
            if lock is not None: lock.release()

    # test data
    if Nstest > 0:
        errT = err_test.T
        if compare is True: errVLT = errVL_test.T
        rPT = lim._rP.T

        errmean = np.zeros(jmax)
        if compare is True: errmeanVL = np.zeros(jmax)
        for i in range(0,jmax):
            errmean[i] = _errmean(i, errT, rPT, deltarend)
            if compare is True: errmeanVL[i] = _errmeanVL(i, errVLT, rPT, deltarend)

        if verbose is not False:
            print('for test data')
            print('meanerr: %s' % errmean.tolist())
        if outfiles is True:
            if lock is not None: lock.acquire()
            rnd = rng.randint(1e7)
            ol = rootdir+'/error_mean_bin_testset_%sxCG_%s_%s.mat' % (CG, mnu, version)
            fl = os.path.splitext(ol)[0]
            sio.savemat(fl+'_%s.mat' % rnd, {'errmean':errmean})
            os.renames(fl+'_%s.mat' % rnd, ol)
            if lock is not None: lock.release()
        if compare is True:
            if verbose is not False: print('meanerrVL: %s' % errmeanVL.tolist())
            if outfiles is True:
                if lock is not None: lock.acquire()
                rnd = rng.randint(1e7)
                ol = rootdir+'/error_meanVL_bin_testset_%sxCG_%s_%s.mat' % (CG, mnu, version)
                fl = os.path.splitext(ol)[0]
                sio.savemat(fl+'_%s.mat' % rnd, {'errmean':errmeanVL})
                os.renames(fl+'_%s.mat' % rnd, ol)
                if lock is not None: lock.release()
    #XXX: return results object?
    #XXX: pick a better metric?
    #result = errmean.max().tolist()
    result = errmean.mean().tolist()
    #if verbose is not False: print('mean(meanerr): %s' % result)
    return result


if __name__ == '__main__':

    # dataset hyperparameters
    CG = 2 # Coarse graining factor #XXX: don't use if generating data ?
    nu = 0.01 # viscosity #TODO: uncertainty in nu ?

    # learning hyperparmeters
    jmax = 20
    phi0 = 0.0

    kwds = dict(version='vZ', verbose=True, Ns=12, T=800)
    error = analysis([nu, CG, jmax, phi0], **kwds)
    print('mean(meanerr): %s' % error)


# EOF
