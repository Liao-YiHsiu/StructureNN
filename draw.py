#!/usr/bin/python
import sys
import matplotlib.pyplot as plt

def main(argv):
   if len(argv) != 2 :
      sys.exit("Usage: draw.py <file> <dir>")

   filename = argv[1]
   dirname  = argv[2]
   points = eval(open(filename, 'r').read())
   for utt in points.keys():
      fig = plt.figure()
      ax  = fig.add_subplot(111)
      ax.plot(points[utt][0], points[utt][1], 'ro')
      fig.savefig('{}/{}.png'.format(dirname, utt))

if __name__ == "__main__":
   main(sys.argv[1:])
