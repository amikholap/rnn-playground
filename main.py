#!/usr/bin/env python3
import argparse

import tensorflow as tf

import rnntf


tf.logging.set_verbosity(tf.logging.WARN)


def main():
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


def make_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    reber = subparsers.add_parser('reber')
    reber.add_argument('--estimator', choices=['lstm', 'transformer'], required=True)
    reber.add_argument('--batch-size', type=int, default=256)
    reber.add_argument('--hidden-units', type=int, default=4)
    reber.add_argument('--dataset-path', type=argparse.FileType('r'))
    reber.add_argument('--embedded', action='store_true')
    reber.set_defaults(func=rnntf.reber.run.run)

    for subparser in subparsers.choices.values():
        subparser.add_argument('--learning-rate', type=float, default=0.1)
        subparser.add_argument('--debug', action='store_true')
        subparser.add_argument('--model-dir')

    return parser


if __name__ == '__main__':
    main()
