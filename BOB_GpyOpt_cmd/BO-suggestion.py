# -*- coding: utf-8 -*-

import sys, getopt
import os.path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt
from GPyOpt.methods import BayesianOptimization


# NOTE: this python script should be in the same folder as input files
# BO-suggestion.py -x prior_RGB.txt -y prior_valvetime.txt -o out_ValveTime.txt
def main(argv):
    input_xfile = ''
    input_yfile = ''
    output_ofile = ''
    try:
        opts, args = getopt.getopt(argv, "h:x:y:o:",
                                   ["x_file=", "y_file=", "o_file="])
    except getopt.GetoptError:
        print('BO-suggestion.py -x <prior_valvetime.txt> -y <prior_RGB.txt> \
        -o <out_ValveTime.txt>')
        sys.exit()

    for opt, arg in opts:
        if opt == '-h':
            print('BO-suggestion.py -x <prior_RGB.txt> \
            -y <prior_valvetime.txt> -o <out_ValveTime.txt>')
            sys.exit()
        elif opt in ("-x", "--x_file"):
            input_Xfile = arg
        elif opt in ("-y", "--y_file"):
            input_Yfile = arg
        elif opt in ("-o", "--o_file"):
            output_Ofile = arg

    return input_xfile, input_yfile, output_ofile


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
elif not os.path.isfile(infileX) :
    print('Input file 1:', infileX, ' is not found.')
    sys.exit()
elif not os.path.isfile(infileY):
    print('Input file 2:', infileY, ' is not found.\n')
    sys.exit()

prior_valvetime_df = pd.read_csv(infileX, sep=",", comment='#', header=None)
prior_valvetime_df.columns = ['tA', 'tB', 'tC']
prior_valvetime = prior_valvetime_df.values
prior_exp_RGB_df = pd.read_csv(infileY, sep=",", comment='#', header=None)
prior_exp_RGB_df.columns = ['R', 'G', 'B']
prior_exp_RGB = prior_exp_RGB_df.values
print(prior_valvetime_df)
print(prior_exp_RGB_df)
if prior_valvetime == np.array([]) or prior_exp_RGB == np.array([]):
    print('One of the input files are empty! Abort!')
    sys.exit()
print(infileX, ' and ', infileY,
      'data exist! Running BO for next acquisition')


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


# Start the BO
Batch_size = 6  # Can be modified based on experiment
RGB_ref = np.array([231, 83, 191])  # Should be modified based on experiment

X = prior_valvetime
Y = mse_func(prior_exp_RGB, RGB_ref)
optimizer = optimizer_func(X, Y, Batch_size)
next_acquisition = optimizer.suggest_next_locations()

next_acquisition_df = pd.DataFrame(data = next_acquisition)
if outfile == '':
    outfile = 'next_acquisition_valvetime.txt'
next_acquisition_df.to_csv(outfile, sep=",", index=False, header=False)
print('Output can be found in\'', outfile)
print(next_acquisition_df)

if __name__ == "__main__":
    infileX, infileY, outfile = main(sys.argv[1:])