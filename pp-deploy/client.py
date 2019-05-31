from sys import path
from os.path import expanduser
#path.append('/home/ubuntu/StatisticalClearSky/')
path.append('/Users/bennetmeyers/Documents/ClearSky/StatisticalClearSky/')
from statistical_clear_sky.algorithm.iterative_fitting import IterativeFitting
from solardatatools import standardize_time_axis, make_2d, fix_time_shifts
import pp
import s3fs
import pandas
import numpy
import sys

from statistical_clear_sky.algorithm.time_shift.clustering\
import ClusteringTimeShift
from\
 statistical_clear_sky.algorithm.initialization.singular_value_decomposition\
 import SingularValueDecomposition
from statistical_clear_sky.algorithm.initialization.linearization_helper\
 import LinearizationHelper
from statistical_clear_sky.algorithm.initialization.weight_setting\
 import WeightSetting
from statistical_clear_sky.algorithm.exception import ProblemStatusError
from statistical_clear_sky.algorithm.minimization.left_matrix\
 import LeftMatrixMinimization
from statistical_clear_sky.algorithm.minimization.right_matrix\
 import RightMatrixMinimization
from statistical_clear_sky.algorithm.serialization.state_data import StateData
from statistical_clear_sky.algorithm.serialization.serialization_mixin\
 import SerializationMixin
from statistical_clear_sky.algorithm.plot.plot_mixin import PlotMixin

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
    # Weird quirk of pp is that some packages such as pandas an numpy need to use the full name for the import
    meta = pandas.read_csv(base + 'sys_meta.csv')
    id = meta['ID'][n]
    # full name
    df = pandas.read_csv(base+'PVOutput/{}.csv'.format(id), index_col=0,
                      parse_dates=[0], usecols=[1, 3])
    tz = meta['TimeZone'][n]
    df.index = df.index\
        .tz_localize(tz)\
        .tz_convert('Etc/GMT+{}'.format(TZ_LOOKUP[tz]))\
        .tz_localize(None)   # fix daylight savings
    data = fix_time_shifts(make_2d(standardize_time_axis(df), key='Power(W)'), verbose=False, c1=5.)
    if verbose:
        print(n, id)
    return data


def progress(count, total, status=''):
    """
    Python command line progress bar in less than 10 lines of code.

    https://gist.github.com/vladignatyev/06860ec2040cb497f0f3

    :param count: the current count, int
    :param total: to total count, int
    :param status: a message to display
    :return:
    """
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()


def calc_deg(n):
    power_signals_d = load_sys(n, verbose=False)
    scsf = IterativeFitting(power_signals_d, rank_k=6, solver_type='MOSEK',
                            auto_fix_time_shifts=False)
    try:
        scsf.execute(mu_l=5e2, mu_r=1e3, tau=0.85, max_iteration=10,
                     exit_criterion_epsilon=5e-3,
                     max_degradation=None, min_degradation=None,
                     verbose=False)
    except:
        output = {
            'deg': numpy.nan,                             # note full name for numpy import
            'res-median': numpy.nan,
            'res-var': numpy.nan,
            'res-L0norm': numpy.nan,
            'solver-error': numpy.nan,
            'f1-increase': numpy.nan,
            'obj-increase': numpy.nan
        }
    else:
        try:
            deg = numpy.float(scsf.beta_value.item())
        except:
            deg = numpy.nan
        output = {
            'deg':deg,                             # note full name for numpy import
            'res-median': scsf.residuals_median,
            'res-var': scsf.residuals_variance,
            'res-L0norm': scsf.residual_l0_norm,
            'solver-error': scsf.state_data.is_solver_error,
            'f1-increase': scsf.state_data.f1_increase,
            'obj-increase': scsf.state_data.obj_increase
        }
    return output


def main(ppservers, pswd, fn, partial=True):
    if partial:
        start = 150
        stop = start + 2
        file_indices = range(start, stop)
    else:
        file_indices = range(573)
    # set ncpus=0 so that only remote nodes work on the problem
    if len(ppservers) > 0 :
        ncpu = 0
    else:
        ncpu = 1
    job_server = pp.Server(ncpus=ncpu, ppservers=ppservers, secret=pswd)
    jobs = [
        (
            ind,
            job_server.submit(
                calc_deg,
                (ind,),
                (load_sys, IterativeFitting, fix_time_shifts, SerializationMixin,
                 ClusteringTimeShift, SingularValueDecomposition,
                 LinearizationHelper, WeightSetting, ProblemStatusError,
                 LeftMatrixMinimization, RightMatrixMinimization, StateData,
                 PlotMixin),
                ("import pandas", "import numpy")
            )
        )
        for ind in file_indices
    ]

    output = pandas.DataFrame(columns=['deg', 'res-median', 'res-var', 'res-L0norm', 'solver-error', 'f1-increase',
                                   'obj-increase', 'fix-ts'])


    num = len(jobs)
    it = 0
    progress(it, num, status='processing files')
    for ind, job in jobs:
        output.loc[ind] = job()
        it += 1
        progress(it, num, status='processing files')
    progress(it, num, status='complete              ')
    # See this SO post for info on how to write csv directly to s3 bucket:
    # https://stackoverflow.com/questions/38154040/save-dataframe-to-csv-directly-to-s3-python
    home = expanduser('~')
    with open(home + '/.aws/credentials') as f:
        lns = f.readlines()
        key = lns[1].split(' = ')[1].strip('\n')
        secret = lns[2].split(' = ')[1].strip('\n')
    bytes_to_write = output.to_csv(None).encode()
    fs = s3fs.S3FileSystem(key=key, secret=secret)
    with fs.open('s3://pvinsight.nrel/output/' + fn + '.csv', 'wb') as f:
        f.write(bytes_to_write)
    print('\n')
    job_server.print_stats()



if __name__ == "__main__":
    num_nodes = input('How many nodes in the cluster? ')
    if int(num_nodes) > 0:
        ips = []
        for it in range(int(num_nodes)):
            ip = input('Enter IP address for node {}: '.format(it + 1))
            ips.append(ip)
        ppservers = tuple((ip + ':35000' for ip in ips))
        pswd = input('What is the password? ')
    else:
        ppservers = ()
        pswd = None
    fn = input('Output file name? ')
    main(ppservers, pswd, fn)
