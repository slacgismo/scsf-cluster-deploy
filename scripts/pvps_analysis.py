
from os.path import expanduser
from sys import path
path.append('/home/ubuntu/StatisticalClearSky/')
#path.append('/Users/bennetmeyers/Documents/ClearSky/StatisticalClearSky/')
from statistical_clear_sky import IterativeFitting
from solardatatools import DataHandler
from solardatatools.utilities import progress
import s3fs
import pandas as pd
import numpy as np
import sys
import pickle
from glob import glob
from time import time

GROUPS = {
    1: [0, 1, 2],
    2: [4, 5, 14],
    3: [6, 7],
    4: [8, 10, 20],
    5: [9],
    6: [12, 13, 26],
    7: [3, 11],
}
DATA_DIR = '/home/ubuntu/data/'
#DATA_DIR = '/Users/bennetmeyers/Documents/PVInsight/IEA_PVPC_Task13/'

POWER_PATTERNS = [
    'power',
    'pdc',
    'p_dc',
    'pac',
    'p_ac',
    'pmp',
    'p_mp',
    '[w]',
    '(w)'
]
IRR_PATTERNS = [
    'gplan',
    'poa',
    'w/m2'
]

def load_table(sys_id, data_dir=DATA_DIR, power_patterns=POWER_PATTERNS,
               irr_patterns=IRR_PATTERNS):
    df = pd.read_hdf(data_dir + 'pvps_data.h5', key='site{:02}'.format(sys_id))
    irrad_cols = [col for col in df.columns if
                  np.any([s in col.lower() for s in irr_patterns])]
    power_cols = [col for col in df.columns if
                  np.any([s in col.lower() for s in power_patterns])]
    use_cols = np.concatenate([irrad_cols, power_cols])
    return df, use_cols

def get_group_total(group_id, summary_table):
    gp = GROUPS[group_id]
    count = np.sum(summary_table.loc[gp].num_cols)
    return count

def main(group_id, data_dir=DATA_DIR):
    sys_ids = GROUPS[group_id]
    analysis_output = pd.DataFrame(columns=[
        'sys_id', 'col_id', 'deg_avg', 'deg_g_ub', 'deg_g_lb', 'deg_med',
        'deg_p_ub', 'deg_p_lb'
    ])
    summary_table = pd.read_csv(data_dir + 'pvps_summary_v2.csv', index_col=0)
    total = get_group_total(group_id, summary_table)
    count = 0
    cache_list = glob('*.scsf')
    ti = time()
    for sys_id in sys_ids:
        df, use_cols = load_table(sys_id)
        dh = DataHandler(df)
        for j, col in enumerate(use_cols):
            tn = time()
            progress(count, total, status=' {:.2f} min {} {}'.format(
                (tn - ti) / 60, sys_id, col
            ))
            dh.run_pipeline(use_col=col, verbose=False)
            if 'pvps{:02}_{}.scsf'.format(sys_id, j) in cache_list:
                scsf = IterativeFitting.load_instance('pvps{:02}_{}.scsf'.format(sys_id, j))
            else:
                scsf = IterativeFitting(data_handler_obj=dh, rank_k=6,
                                    solver_type='MOSEK')
            try:
                scsf.execute(max_iteration=20, non_neg_constraints=False, mu_l=1e5,
                             bootstraps=50, exit_criterion_epsilon=5e-3, verbose=False)
            except:
                deg_avg = np.nan
                deg_g_ub = np.nan
                deg_g_lb = np.nan
                deg_med = np.nan
                deg_p_ub = np.nan
                deg_p_lb = np.nan
            else:
                scsf.save_instance('pvps{:02}_{}.scsf'.format(sys_id, j))
                with open('pvps{:02}_{}_bootstraps.pkl'.format(sys_id, j),
                          'wb+') as f:
                    pickle.dump(dict(scsf.bootstrap_samples), f)
                bootstrap_samples = dict(scsf.bootstrap_samples)
                bootbetas = [val['beta'] for val in bootstrap_samples.values()]
                deg_avg = 100 * np.average(bootbetas)
                deg_g_ub = deg_avg + 100 * 1.96 * np.std(bootbetas)
                deg_g_lb = deg_avg - 100 * 1.96 * np.std(bootbetas)
                deg_med = 100 * np.median(bootbetas)
                deg_p_ub = np.percentile(bootbetas, 97.5)
                deg_p_lb = np.percentile(bootbetas, 2.5)
            analysis_output.loc[count] = [
                sys_id, j, deg_avg, deg_g_ub, deg_g_lb, deg_med,
                deg_p_ub, deg_p_lb
            ]
            count += 1
    analysis_output.to_csv('pvps_deg_analysis_batch_{}'.format(group_id))
    tn = time()
    progress(count, total, status=' {:.2f} min DONE!          '.format(
                (tn - ti) / 60
            ))

if __name__ == "__main__":
    group_id = int(sys.argv[1])
    main(group_id)