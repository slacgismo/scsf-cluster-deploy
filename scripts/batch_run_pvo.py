"""Command-line Script for SCSF on PVO Data

This script provides a command-line interface for breaking up a fleet-scale
analysis using Statistical Clear Sky Fitting (SCSF).

Basic Usage:

    python batch_run_pvo.py [worker #] [total # of workers]

Example:

    python batch_run_pvo.py 4 20

    Would start the 4th worker of a cluster containing 20 EC2 compute instances.

AWS Implementation:

    Begin by setting up a single EC2 instance with all necessary code:

        - SCSF
        - solar-data-tools
        - Additional imports listed below
        - This script

    Test the code stack on a single instance. Then, create an AMI of that instance
    in the AWS console. Spin up as many instances as you want in your cluster
    from that image. For each EC2 instance, do the following:

        (1) ssh to instance
        (2) start a screen
        (3) (optional but recommended) start virtual environment
        (4) start script
        (5) detach screen
        (6) close ssh connection

    These step correspond with the following commands/keystrokes

        (1) ssh -i ~/.aws/bennetm-slac-gismo.pem ubuntu@[aws-server-location].compute.amazonaws.com
        (2) screen
        (3) workon scsf # for a virtual environment named "scsf"
        (4) python batch_run_pvo.py [worker #] [total # of workers]
        (5) ctr-a d     # keystroke
        (6) exit

Remarks:

    This script will not perform dynamic load balancing, but it does attempt to
    improve the efficiency of the cluster by randomizing the selection of the
    batches for the workers. A random seed is used to ensure consistency across
    the worker nodes such that all files are processed exactly once. There can
    be a large variance in the amount of time it takes to analyze a single site,
    depending on how long the data set is, how extreme the shade pattern is,
    and generally how difficult the data is to fit the to SCSF model.
    Additionally, many fleet-scale data sets have highly correlated data streams
    which can cluster in the ordering of the sites, by name or ID. Randomization
    helps remove clusters of data streams that all take around the same amount
    of time to fit with SCSF, mitigating very slow and very fast worker nodes.

"""


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

def calc_deg(n, meta):
    power_signals_d, sys_id = load_sys(n, meta, verbose=VERBOSE)
    scsf = IterativeFitting(power_signals_d, rank_k=6, solver_type='MOSEK',
                            auto_fix_time_shifts=False)
    try:
        scsf.execute(mu_l=5e2, mu_r=1e3, tau=0.85, max_iteration=10,
                     exit_criterion_epsilon=5e-3,
                     max_degradation=None, min_degradation=None,
                     verbose=VERBOSE)
    except:
        output = {
            'sys_id': sys_id,
            'deg': numpy.nan,                             # note full name for numpy import
            'res-median': numpy.nan,
            'res-var': numpy.nan,
            'res-L0norm': numpy.nan,
            'solver-error': True,
            'f1-increase': numpy.nan,
            'obj-increase': numpy.nan
        }
    else:
        try:
            deg = numpy.float(scsf.beta_value.item())
        except:
            deg = numpy.nan
        output = {
            'sys_id': sys_id,
            'deg':deg,                             # note full name for numpy import
            'res-median': scsf.residuals_median,
            'res-var': scsf.residuals_variance,
            'res-L0norm': scsf.residual_l0_norm,
            'solver-error': scsf.state_data.is_solver_error,
            'f1-increase': scsf.state_data.f1_increase,
            'obj-increase': scsf.state_data.obj_increase
        }
    return output

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

def main(num_workers, worker_id, fn):
    metadata = read_metadata()
    num_systems = len(metadata)
    numpy.random.seed(42)
    index_order = numpy.arange(num_systems)
    numpy.random.shuffle(index_order)
    splits = numpy.array_split(index_order, num_workers)
    sys_index = splits[worker_id - 1]

    output = pandas.DataFrame(columns=['sys_id', 'deg', 'res-median',
                                       'res-var', 'res-L0norm', 'solver-error',
                                       'f1-increase', 'obj-increase'])
    count = 0
    total_count = len(sys_index)
    ti = time()
    for n in sys_index:
        progress(count, total_count, 'processing #{}'.format(n))
        row = calc_deg(n, metadata)
        output.loc[n] = row
        count += 1
    tf = time()
    msg = 'done! {:.2f} miniutes                      '.format((tf - ti) / 60.)
    progress(count, total_count, msg)

    # See this SO post for info on how to write csv directly to s3 bucket:
    # https://stackoverflow.com/questions/38154040/save-dataframe-to-csv-directly-to-s3-python
    home = expanduser('~')
    with open(home + '/.aws/credentials') as f:
        lns = f.readlines()
        key = lns[1].split(' = ')[1].strip('\n')
        secret = lns[2].split(' = ')[1].strip('\n')
    bytes_to_write = output.to_csv(None).encode()
    fs = s3fs.S3FileSystem(key=key, secret=secret)
    file_path = 's3://pvinsight.nrel/output/' + fn + '_batch{:02}'.format(worker_id) + '.csv'
    with fs.open(file_path, 'wb') as f:
        f.write(bytes_to_write)
    return


if __name__ == "__main__":
    num_workers = int(sys.argv[2])
    worker_id = int(sys.argv[1])
    main(num_workers, worker_id, FILENAME)
