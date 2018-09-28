#!/usr/bin/env python3
import argparse

import rnntf


def main():
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)


def make_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    reber_lstm = subparsers.add_parser('reber_lstm')
    reber_lstm.add_argument('--batch-size', type=int, default=128)
    reber_lstm.add_argument('--hidden-size', type=int, default=4)
    reber_lstm.add_argument('--dataset-path', type=argparse.FileType('r'))
    reber_lstm.add_argument('--embedded', action='store_true')
    reber_lstm.set_defaults(func=rnntf.reber.lstm.run)

    return parser


if __name__ == '__main__':
    main()
