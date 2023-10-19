import os
import math
import argparse
import numpy as np
from tabulate import tabulate
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res-dir', type=str, default='', help='Path to the results')
    parser.add_argument('--split-list', type=int, nargs='+', default=[1], help='')
    parser.add_argument('--shot-list', type=int, nargs='+', default=[10], help='')

    args = parser.parse_args()

    wf = open(os.path.join(args.res_dir, 'results.txt'), 'w')
    shot = args.shot
    dir_list = [os.path.join('defrcn_det_r101_base4')]
    dir_list += [os.path.join(args.res_dir, ['defrcn_ifsod_r101_novel{}'.format(split), 'tfa-like', '{}shot_seed0'.format(shot)]) for split in args.split_list]

    results = []
    header = ['sid', 'base', 'bird', 'bus', 'cow', 'motorbike', 'sofa']
    for _dir in dir_list:
        fpath = os.path.join(_dir, 'log.txt')
        lineinfos = open(fpath).readlines()
        mAPs = [float(x.strip()) for x in lineinfos[-1].strip().split('|')]
        for i in range(20 - len(mAPs)):
            mAPs.append(-1)
        mAPs_np = np.array(mAPs)
        results.append(mAPs_np[:15].mean())

        for fid, fpath in enumerate(sorted(file_paths)):
            lineinfos = open(fpath).readlines()
            if fid == 0:
                res_info = lineinfos[-2].strip()
                header = res_info.split(':')[-1].split(',')
            res_info = lineinfos[-1].strip()
            results.append([fid] + [float(x) for x in res_info.split(':')[-1].split(',')])

    results_np = np.array(results)
    avg = np.mean(results_np, axis=0).tolist()
    cid = [1.96 * s / math.sqrt(results_np.shape[0]) for s in np.std(results_np, axis=0)]
    results.append(['μ'] + avg[1:])
    results.append(['c'] + cid[1:])

    table = tabulate(
        results,
        tablefmt="pipe",
        floatfmt=".2f",
        headers=[''] + header,
        numalign="left",
    )

    wf.write('--> {}-shot\n'.format(shot))
    wf.write('{}\n\n'.format(table))
    wf.flush()
    wf.close()

    print('Reformat all results -> {}'.format(os.path.join(args.res_dir, 'results.txt')))


if __name__ == '__main__':
    main()
