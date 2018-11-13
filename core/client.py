from sys import path
from os.path import expanduser
#path.append('/home/ubuntu/StatisticalClearSky/')
path.append('/Users/bennetmeyers/Documents/ClearSky/StatisticalClearSky/')
from clearsky.main import IterativeClearSky, ProblemStatusError, fix_time_shifts
from clearsky.utilities import CONFIG1
import pp
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
    # Weird quirk of pp is that some packages such as pandas an numpy need to use the full name for the import
    meta = pandas.read_csv(base + 'sys_meta.csv')
    id = meta['ID'][n]
    # full name
    df = pandas.read_csv(base+'PVOutput/{}.csv'.format(id), index_col=0,
                      parse_dates=[0], usecols=[1, 3])
    tz = meta['TimeZone'][n]
    df.index = df.index.tz_localize(tz).tz_convert('Etc/GMT+{}'.format(TZ_LOOKUP[tz]))   # fix daylight savings
    start = df.index[0]
    end = df.index[-1]
    # full name
    time_index = pandas.date_range(start=start, end=end, freq='5min')
    df = df.reindex(index=time_index, fill_value=0)
    if verbose:
        print(n, id)
    return df


def progress(count, total, status=''):
    """
    Python command line progress bar in less than 10 lines of code. · GitHub

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


def calc_deg(n, config):
    df = load_sys(n, verbose=False)
    days = df.resample('D').max().index[1:-1]
    start = days[0]
    end = days[-1]
    D = df.loc[start:end].iloc[:-1].values.reshape(288, -1, order='F')
    ics = IterativeClearSky(D, k=4)
    ics.minimize_objective(verbose=False, **config)
    output = {
        'deg': numpy.float(ics.beta.value),                             # note full name for numpy import
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
        start = 150
        stop = start + 2
        file_indices = range(start, stop)
    else:
        file_indices = range(573)
    job_server = pp.Server(ppservers=ppservers, secret=pswd)
    jobs = [
        (
            ind,
            job_server.submit(
                calc_deg,
                (ind, CONFIG1),
                (load_sys, IterativeClearSky, ProblemStatusError, fix_time_shifts),
                ("import pandas", "import numpy")
            )
        )
        for ind in file_indices
    ]

    output = pd.DataFrame(columns=['deg', 'res-median', 'res-var', 'res-L0norm', 'solver-error', 'f1-increase',
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
            ip = input('Enter IP address for node {}'.format(it + 1))
        ppservers = tuple((ip + ':35000' for ip in ips))
        pswd = input('What is the password? ')
    else:
        ppservers = ()
        pswd = None
    fn = input('Output file name? ')
    main(ppservers, pswd, fn)
