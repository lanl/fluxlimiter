from mystic.solvers import DifferentialEvolutionSolver2
from mystic.solvers import BuckshotSolver, PowellDirectionalSolver
from mystic.solvers import SparsitySolver, NelderMeadSimplexSolver
from mystic.monitors import VerboseLoggingMonitor, Monitor
from mystic.termination import Or, VTR
from mystic.termination import ChangeOverGeneration as COG
from mystic.termination import CandidateRelativeTolerance as CRT
from hfcall_NONequalbins_nonoise import *

def configure(**args):
    "build cost and callback"
    clean = args.pop('clean', True)
    lock = args.pop('lock', None)

    def cost(x, **kwds): #XXX: ignore axis
        import uuid
        arg = args.copy()
        version = arg.pop('version', uuid.uuid4().hex[:7])
        Ns = int(arg.pop('Ns', 120))
        T = arg.pop('T', 800)
        verbose = arg.pop('verbose', None)
        return analysis(x, version=version, verbose=verbose, Ns=Ns, T=T, lock=lock, **arg)

    def callback(x):
        "delete all run files except those corresponding to x"
        nu, CG, jmax, phi0 = x #XXX
        jmax = int(jmax); CG = int(CG)
        mphi = int(phi0 * 1000)
        mnu = int(nu * 1000)
        version = args.get('version', '*') #XXX
        Ns = int(args.get('Ns', 120)) #XXX
        T = args.get('T', 800) #XXX
        verbose = args.get('verbose', None)
        rootdir = 'run'+str(jmax)+'_%s_%sxCG_%s_%s' % (mphi, CG, mnu, version)
        alldirs = 'run*_*_*xCG_*_%s' % version
        allfiles = '6pts_*cg_%sNs_T%s_*mnu_%s.mat' % (Ns, T, version)
        inputfile = "6pts_%scg_%sNs_T%s_%smnu_%s.mat" % (CG, Ns, T, mnu, version)
        import os
        import glob
        import pox
        rootdir = glob.glob(rootdir)
        inputfile = glob.glob(inputfile)
        # move files that should be saved
        for path in rootdir:
            if verbose is not False: print('skipping: %s' % path)
            #os.renames(path, '_'+path)
        for path in inputfile:
            if verbose is not False: print('skipping: %s' % path)
            #os.renames(path, '_'+path)
        # remove all other files
        alldirs = (r for r in glob.glob(alldirs) if r not in rootdir)
        allfiles = (f for f in glob.glob(allfiles) if f not in inputfile)
        for path in alldirs: #FIXME: conflict/extraneous when same 'x'
            if verbose is not False: print('removing: %s' % path)
            pox.rmtree(path, True, True)
        for path in allfiles:
            if verbose is not False: print('removing: %s' % path)
            os.remove(path)
        # return files to proper names
        #for path in rootdir:
        #    os.renames('_'+path, path)
        #for path in inputfile:
        #    os.renames('_'+path, path)
        return
    return cost, (callback if clean is True else None)


def solve(objective, lb, ub, **kwds):
    solver = kwds.get('solver', DifferentialEvolutionSolver2)
    npop = kwds.get('npop', None)
    if npop is not None:
        solver = solver(len(lb),npop)
    else:
        solver = solver(len(lb))
    solver.id = kwds.pop('id', None)
    nested = kwds.get('nested', None)
    x0 = kwds.get('x0', None)
    if nested is not None: # Buckshot/Sparsity
        solver.SetNestedSolver(nested)
    else: # DiffEv/Nelder/Powell
        if x0 is None: solver.SetRandomInitialPoints(min=lb,max=ub)
        else: solver.SetInitialPoints(x0)
    save = kwds.get('save', None)
    if save is not None:
        solver.SetSaveFrequency(save, 'Solver.pkl') #XXX: set name?
    mapper = kwds.get('pool', None)
    if mapper is not None:
        pool = mapper() #XXX: ThreadPool, ProcessPool, etc
        solver.SetMapper(pool.map) #NOTE: not Nelder/Powell
    maxiter = kwds.get('maxiter', None)
    maxfun = kwds.get('maxfun', None)
    solver.SetEvaluationLimits(maxiter,maxfun)
    evalmon = kwds.get('evalmon', None)
    evalmon = Monitor() if evalmon is None else evalmon
    solver.SetEvaluationMonitor(evalmon[:0])
    stepmon = kwds.get('stepmon', None)
    stepmon = Monitor() if stepmon is None else stepmon
    solver.SetGenerationMonitor(stepmon[:0])
    solver.SetStrictRanges(min=lb,max=ub)
    constraints = kwds.get('constraints', None)
    if constraints is not None:
        solver.SetConstraints(constraints)
    opts = kwds.get('opts', {})
    # solve
    solver.Solve(objective, **opts)
    if mapper is not None:
        pool.close()
        pool.join()
        pool.clear() #NOTE: if used, then shut down pool
    #NOTE: debugging code
    #print("solved: %s" % solver.Solution())
    #func_bound = solver.bestEnergy
    #func_evals = solver.evaluations
    #from mystic.munge import write_support_file
    #write_support_file(solver._stepmon)
    #print("func_bound: %s" % func_bound) #NOTE: may be inverted
    #print("func_evals: %s" % func_evals)
    return solver


if __name__ == '__main__':

    import multiprocess as mp
    import pathos.pools as pp

    #lock = mp.Manager().Lock()
    lock = None #FIXME: Ns=20?; use default version
    settings = dict(verbose=None, Ns=2, T=800, clean=True, version='nolk')
    cost, callback = configure(lock=lock, **settings)

    # kwds for solver
    opts = dict(termination=COG(1e-10, 100), callback=callback)
    param = dict(solver=DifferentialEvolutionSolver2,
                 npop=2,#FIXME: 10,
                 maxiter=1,#FIXME: 1500,
                 maxfun=1e+6,
                 x0=None, # use RandomInitialPoints
                 nested=None, # don't use SetNested
                 save=1, # save solver every iteration
                 #save=None, # don't save Solver
                 pool=pp.ProcessPool,
                 #pool=None, # don't use SetMapper
                 stepmon=VerboseLoggingMonitor(1,1,1,label='output'),
                 evalmon=Monitor(), # monitor config (re-initialized in solve)
                 # kwds to pass directly to Solve(objective, **opt)
                 opts=opts,
                )

    # hyperparam: [nu, CG, jmax, phi0]
    lb = [1e-3, 1, 10, 0]
    ub = [1e-1, 5, 30, 0]

    from mystic.constraints import integers

    @integers(ints=float, index=(1,2))
    def constrain(x):
        return x

    solver = solve(cost, lb, ub, constraints=constrain, **param)

    print("solved: %s" % solver.Solution())
    func_bound = solver.bestEnergy
    func_evals = solver.evaluations
    from mystic.munge import write_support_file
    write_support_file(solver._stepmon)
    print("func_bound: %s" % func_bound)
    print("func_evals: %s" % func_evals)

# EOF
