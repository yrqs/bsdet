import torch
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-path', type=str, default='', help='Path to the main checkpoint')

    args = parser.parse_args()

    ckpt = torch.load(args.src_path)
    if 'scheduler' in ckpt:
        del ckpt['scheduler']
    if 'optimizer' in ckpt:
        del ckpt['optimizer']
    if 'iteration' in ckpt:
        ckpt['iteration'] = 0
    torch.save(ckpt, args.src_path)