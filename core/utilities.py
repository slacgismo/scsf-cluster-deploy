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