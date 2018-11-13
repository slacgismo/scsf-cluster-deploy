from sys import path, argv
path.append('/home/ubuntu/StatisticalClearSky/')
from clearsky.main import IterativeClearSky, ProblemStatusError, fix_time_shifts
from clearsky.utilities import CONFIG1
import pp
import numpy as np
import s3fs
import pandas as pd
import sys

TZ_LOOKUP = {
    'America/Anchorage': 9,
    'America/Chicago': 6,
    'America/Denver': 7,
    'America/Los_Angeles': 8,
    'America/New_York': 5,
    'America/Phoenix': 7,
    'Pacific/Honolulu': 10
}

def load_sys(n, fp=None, verbose=False):
    if fp is not None:
        base = fp
    else:
        base = 's3://pvinsight.nrel/PVO/'
    try:
        meta = pd.read_csv('local_data/sys_meta.csv')
    except FileNotFoundError:
        meta = pd.read_csv(base + 'sys_meta.csv')
        meta.to_csv('local_data/sys_meta.csv')
    id = meta['ID'][n]
    df = pd.read_csv(base+'PVOutput/{}.csv'.format(id), index_col=0,
                      parse_dates=[0], usecols=[1, 3])
    tz = meta['TimeZone'][n]
    df.index = df.index.tz_localize(tz).tz_convert('Etc/GMT+{}'.format(TZ_LOOKUP[tz]))   # fix daylight savings
    start = df.index[0]
    end = df.index[-1]
    time_index = pd.date_range(start=start, end=end, freq='5min')
    df = df.reindex(index=time_index, fill_value=0)
    if verbose:
        print(n, id)
    return df


def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


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
                (load_sys,),
                ('from clearsky.main import IterativeClearSky, ProblemStatusError, fix_time_shifts',
                 'import pandas as pd', 'import numpy as np', 'from numpy.linalg import norm', 'import cvxpy as cvx',
                 'from time import time', 's3fs')
            )
        )
        for ind in file_indices
    ]
    output = pd.DataFrame(columns=['deg', 'res-median', 'res-var', 'res-L0norm', 'solver-error', 'f1-increase',
                                   'obj-increase', 'fix-ts'])
    num = len(jobs)
    it = 0
    for ind, job in jobs:
        output.loc[ind] = job()
        progress(it, num, status='processing files')
        it += 1
    progress(it, num, status='complete')
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
