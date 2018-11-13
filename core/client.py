from sys import path, argv
path.append('/home/ubuntu/StatisticalClearSky/')
from clearsky.main import IterativeClearSky, ProblemStatusError, fix_time_shifts
from clearsky.utilities import CONFIG1
from utilities import load_sys, progress
import pp
import numpy as np
import pandas as pd
import s3fs


def calc_deg(n, config):
    df = load_sys(n, verbose=False)
    days = df.resample('D').max().index[1:-1]
    start = days[0]
    end = days[-1]
    D = df.loc[start:end].iloc[:-1].values.reshape(288, -1, order='F')
    ics = IterativeClearSky(D, k=4)
    ics.minimize_objective(**config)
    output = {
        'deg': np.float(ics.beta.value),
        'res-median': ics.residuals_median,
        'res-var': ics.residuals_variance,
        'res-L0norm': ics.residual_l0_norm,
        'solver-error': ics.isSolverError,
        'f1-increase': ics.f1Increase,
        'obj-increase': ics.objIncrease,
        'fix-ts': ics.fixedTimeStamps
    }
    return output

def main(ppservers, pswd, fn, partial=True):
    if partial:
        file_indices = range(200, 204)
    else:
        file_indices = range(573)
    job_server = pp.Server(ncpus=1, ppservers=ppservers, secret=pswd)
    jobs = [
        (
            ind,
            job_server.submit(
                calc_deg,
                (ind, CONFIG1),
                (load_sys, IterativeClearSky, ProblemStatusError, fix_time_shifts),
                ('import pandas as pd', 'import numpy as np', 'from numpy.linalg import norm', 'import cvxpy as cvx',
                 'from time import time', 's3fs')
            )
        )
        for ind in file_indices
    ]
    output = pd.DataFrame(columns=['deg', 'res-median', 'res-var', 'res-L0norm', 'solver-error', 'f1-increase',
                                   'obj-increase', 'fix-ts'])
    num = len(jobs)
    for ind, job in jobs:
        output.loc[ind] = job()
        progress(ind+1, num, status='processing files')
    output.to_csv( 's3://pvinsight.nrel/output/' + fn)



if __name__ == "__main__":
    num_nodes = input('How many nodes in the cluster? ')
    if int(num_nodes) > 0:
        ips = []
        for it in range(int(num_nodes)):
            ip = input('Enter IP address for node {}'.format(it + 1))
        ppservers = tuple((ip + ':35000' for ip in ips))
        pswd = input('What is the password? ')
    else:
        ppservers = ()
        pswd = None
    fn = input('Output file name? ')
    main(ppservers, pswd, fn)
