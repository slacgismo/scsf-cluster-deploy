from os.path import expanduser
from sys import path
#path.append('/home/ubuntu/StatisticalClearSky/')
path.append('/Users/bennetmeyers/Documents/ClearSky/StatisticalClearSky/')
from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting
from solardatatools import standardize_time_axis, make_2d, fix_time_shifts
import s3fs
import pandas
import numpy
import sys
from time import time


FILENAME = 'PVO_run_20190530'
VERBOSE = False


TZ_LOOKUP = {
    'America/Anchorage': 9,
    'America/Chicago': 6,
    'America/Denver': 7,
    'America/Los_Angeles': 8,
    'America/New_York': 5,
    'America/Phoenix': 7,
    'Pacific/Honolulu': 10
}

def read_metadata(fp=None):
    if fp is not None:
        base = fp
    else:
        base = 's3://pvinsight.nrel/PVO/'
    # Weird quirk of pp is that some packages such as pandas an numpy need to use the full name for the import
    meta = pandas.read_csv(base + 'sys_meta.csv')
    return meta

def load_sys(n, meta, fp=None, verbose=VERBOSE):
    if fp is not None:
        base = fp
    else:
        base = 's3://pvinsight.nrel/PVO/'
    # Weird quirk of pp is that some packages such as pandas an numpy need to use the full name for the import
    sys_id = meta['ID'][n]
    # full name
    df = pandas.read_csv(base+'PVOutput/{}.csv'.format(sys_id), index_col=0,
                      parse_dates=[0], usecols=[1, 3])
    # Fix daylight savings time shifts
    tz = meta['TimeZone'][n]
    df.index = df.index\
        .tz_localize(tz)\
        .tz_convert('Etc/GMT+{}'.format(TZ_LOOKUP[tz]))\
        .tz_localize(None)   # fix daylight savings
    # Standard solar-data-tools pre-processing steps
    data = fix_time_shifts(make_2d(standardize_time_axis(df), key='Power(W)'), verbose=VERBOSE, c1=5.)
    if verbose:
        print(n, sys_id)
    return data, sys_id

def run_sim(inputs, data, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    rank, mu_l, mu_r, tau, run_number = inputs
    tnow = time()
    iterative_fitting = IterativeFitting(data, rank_k=rank, solver_type='MOSEK')
    iterative_fitting.execute(mu_l=mu_l, mu_r=mu_r, tau=tau, max_iteration=10, verbose=False)
    fn = directory + 'run{:02}.scsf'.format(run_number)
    iterative_fitting.save_instance(fn)
    solve_time = (time() - tnow) / 60
    obj = iterative_fitting.calculate_objective_with_result(sum_components=False)
    degrate = iterative_fitting.beta_value.item()
    row = [run_number, solve_time, rank, mu_l, mu_r, tau, np.sum(obj), obj[0], obj[1], obj[2], degrate]
    return row

def main():
    pass


if __name__ == '__main__':
    pass
