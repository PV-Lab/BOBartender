# -*- coding: utf-8 -*-

import sys
import os.path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from GPyOpt.methods import BayesianOptimization
#import serial
#import time


# # Read and record the data from Ardruino
# # Set up the serial line
# ser = serial.Serial('COM4', 9600)
# time.sleep(2)
# rgb_data =[]                       # empty list to store the data
# for i in range(5):
#     b = ser.readline()         # read a byte string
#     string_n = b.decode()  # decode byte string into Unicode
#     string = string_n.rstrip() # remove \n and \r
#     flt = float(string)        # convert string to float
#     rgb_data.append(flt)           # add to the end of data list
#     time.sleep(0.5)            # wait (sleep) 0.5 seconds


infileX = 'prior_valvetime.txt'
infileY = 'prior_exp_RGB.txt'
outfile = ''

if infileX == '' and infileY == '':
    print('Input <prior_valvetime.txt> and <prior_RGB.txt> are required \
    by option: "-x <prior_RGB.txt> -y <prior_valvetime.txt>" \n')
    sys.exit()
elif infileX == '':
    print('Input <prior_valvetime.txt> is required \
    by option: "-x  <prior_valvetime.txt> "\n')
    sys.exit()
elif infileY == '':
    print('Input <prior_RGB.txt> is required \
    by option: "-y <prior_RGB.txt>"\n')
    sys.exit()

if not os.path.isfile(infileX) and not os.path.isfile(infileY):
    print('Both input file 1:', infileX, 'and Input file 2:',
          infileY, ' are not found.')
    sys.exit()
elif not os.path.isfile(infileX):
    print('Input file 1:', infileX, ' is not found.')
    sys.exit()
elif not os.path.isfile(infileY):
    print('Input file 2:', infileY, ' is not found.\n')
    sys.exit()

prior_valvetime_df = pd.read_csv(infileX, sep=",", comment='#', header=None)
prior_exp_RGB_df = pd.read_csv(infileY, sep=",", comment='#', header=None)

prior_valvetime_df.columns = ['tA', 'tB', 'tC']
prior_valvetime = prior_valvetime_df.values
prior_exp_RGB_df.columns = ['R', 'G', 'B']
prior_exp_RGB = prior_exp_RGB_df.values
print(prior_valvetime_df)
print(prior_exp_RGB_df)
if prior_valvetime == np.array([]) or prior_exp_RGB == np.array([]):
    print('One of the input files are empty! Abort!')
    sys.exit()
print('Running BO for next acquisition')


def optimizer_func(x, y, batch_size):
    bds = [{'name': 'x1', 'type': 'continuous', 'domain': (0, 1.0)},
           {'name': 'x2', 'type': 'continuous', 'domain': (0, 1.0)},
           {'name': 'x3', 'type': 'continuous', 'domain': (0, 1.0)},
           ]
    # kernel = GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=1.0)
    return BayesianOptimization(f=None,
                                domain=bds,
                                model_type='GP',
                                acquisition_type='EI',
                                acquisition_jitter=0.01,
                                X=x,
                                Y=y,
                                evaluator_type='local_penalization',
                                batch_size=batch_size,
                                normalize_Y=True
                                # noise_var = 0.05**2,
                                # kernel = kernel
                                )


def mse_func(rgb_exp, rgb_ref):
    y = []
    for i in range(len(rgb_exp)):
        y.append([mean_squared_error(rgb_exp[i], rgb_ref)])
    return np.array(y)


def plot_rgb_batch(rgb_ref, rgb_exp, title_string):
    plt.figure(1)
    plt.bar(np.arange(1) - 0.25, [rgb_ref[0]], 0.25, alpha=1, color='r')
    plt.bar(np.arange(1) - 0.00, [rgb_ref[1]], 0.25, alpha=1, color='g')
    plt.bar(np.arange(1) + 0.25, [rgb_ref[2]], 0.25, alpha=1, color='b')
    plt.bar(np.transpose(np.arange(1, 7) - 0.25), rgb_exp[:, 0], 0.25,
            alpha=0.5, color='r')
    plt.bar(np.transpose(np.arange(1, 7) + 0.00), rgb_exp[:, 1], 0.25,
            alpha=0.5, color='g')
    plt.bar(np.transpose(np.arange(1, 7) + 0.25), rgb_exp[:, 2], 0.25,
            alpha=0.5, color='b')
    plt.ylabel('RGB values [0 - 256]')
    plt.ylim(0, 300)
    plt.title(title_string)
    plt.show()


def rmse_rgb_3dplot(x, y, title_string):
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=np.squeeze(np.sqrt(y)),
                     linewidth=3, vmin=0, vmax=25)
    cbar = fig.colorbar(img)
    cbar.set_label('RGB RMSE between ref and exp')
    ax.set(xlabel='t_A (sec)', ylabel='t_B (sec)', zlabel='t_C (sec)',
           title=title_string)
    ax.set_xlim3d(0, 1)
    ax.set_ylim3d(0, 1)
    ax.set_zlim3d(0, 1)
    ax.view_init(30, 360 - 60)
    plt.show()


# Start the BO
Batch_size = 6  # Can be modified based on experiment
RGB_ref = np.array([231, 83, 191])  # Should be modified based on experiment

X = prior_valvetime
Y = mse_func(prior_exp_RGB, RGB_ref)
optimizer = optimizer_func(X, Y, Batch_size)
next_acquisition = optimizer.suggest_next_locations()

next_acquisition_df = pd.DataFrame(data=next_acquisition)
if outfile == '':
    outfile = 'next_acquisition_valvetime.txt'
next_acquisition_df.to_csv(outfile, sep=",", index=False, header=False)
print('Output can be found in\'', outfile)
print(next_acquisition_df)

#plot_rgb_batch(RGB_ref, prior_exp_RGB, '')
rmse_rgb_3dplot(prior_valvetime, Y, '')
