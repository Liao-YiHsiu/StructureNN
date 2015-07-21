#!/usr/bin/python
import sys
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def main(argv):
   if len(argv) != 4 :
      sys.exit("Usage: draw.py <file> <dir> <x-label>")

   filename = argv[1]
   dirname  = argv[2]
   xlabel   = argv[3]

   if not os.path.exists(dirname):
           os.makedirs(dirname)

   mpl.rcParams.update({'font.size': 25})

   points = eval(open(filename, 'r').read())

   for utt in points.keys():
      fig = plt.figure()
      ax  = fig.add_subplot(111)
      ax.scatter(points[utt][0], points[utt][1],\
              c=range(len(points[utt][0])), s= 350)
      #fig.suptitle(utt, fontsize=20)
      plt.ylabel('Phoneme Accuracy', fontsize=30)
      plt.xlabel(xlabel, fontsize=30, labelpad=20)
      start, end = ax.get_xlim()
      ax.xaxis.set_ticks(np.arange(start+0.01, end, (end-start-0.000001)/3))
      ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.02f'))

      start, end = ax.get_ylim()
      ax.yaxis.set_ticks(np.arange(start+0.01, end, (end-start-0.000001)/3))
      plt.tight_layout()
      fig.savefig('{}/{}.png'.format(dirname, utt))
      plt.close(fig)

if __name__ == "__main__":
   main(sys.argv)
