''' given a log file, parse it, find the lines which includes the keywork "Inception score mean:". If the line ends with the "after compression.", then extract the numbers after these keyworks: ""Inception score mean:"", and "FID score:" respectively'''
import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import stats

def parse_log(log_file):
    ''' parse the log file, return the list of inception score mean and FID score'''
    with open(log_file, 'r') as f:
        lines = f.readlines()
    inception_score_mean = []
    fid_score = []
    for line in lines:
        if line.find('Inception score mean:') != -1:
            if line.endswith('after compression.\n'):
                inception_score_mean.append(float(line.split('Inception score mean: ')[1].split(',')[0]))#.split('FID score:')[0]))
                fid_score.append(float(line.split('FID score: ')[1].split(' ')[0]))
    return inception_score_mean, fid_score

parser = argparse.ArgumentParser()
parser.add_argument('-l', type=str, default='log.txt', help='log file')
args = parser.parse_args()

inception_score_mean, fid_score = parse_log(args.l)

print('fid score/Inception score:')
print('{} {}'.format(fid_score[0],inception_score_mean[0]))

