#!/usr/bin/env python

"""
Simple data process procedure that converts regular movie rating records into
numpy matrix like data for probabilistic matrix function 
"""
import argparse
import numpy as np
from multiprocessing import Pool
import csv

parser = argparse.ArgumentParser(description="Convert data into binary matrix for numpy processing.")
parser.add_argument('InputTrainfileName',type=str,help="Input Train filename")
parser.add_argument('InputTestfileName', type=str, help="Input Test filename")
parser.add_argument('OutputTrainfileName', type=str, help="Output Train filename")
parser.add_argument('OutputTestfileName', type=str, help="Output Test filename")
args = parser.parse_args()

#list of inputs for different files
inputTrainfilename = args.InputTrainfileName
inputTestfilename = args.InputTestfileName
outputTrainfilename = args.OutputTrainfileName
outputTestfilename = args.OutputTestfileName

"""
Drop datetime information from train row and read other information
"""
def readTrainRow(row):
    return [int(row['movie-id']),int(row['customer-id']),int(row['rating'])]

"""
Drop datetime information from test row and read other information
"""
def readTestRow(row):
    return [int(row['movie-id']),int(row['customer-id'])]


from timeit import default_timer
class Timer(object):
  def __init__(self):
    self.timer = default_timer
    self.start = None
    self.end = None

  def __enter__(self):
    self.start = self.timer()
    self.end = None
    return self

  def __exit__(self, *args):
    self.end = self.timer()

  @property
  def elapsed(self):
    now = self.timer()
    if self.end is not None:
      self.end - self.start
    else:
      return now - self.start

  def rate(self, count):
    now = self.timer()
    if self.start is None:
      raise ValueError("Not yet started")

    return count / (now - self.start)

reader = csv.DictReader(open(inputTestfilename))
pool = Pool()

data = np.empty([1,2],dtype='int')
testUsers = {}
testMovies = {}

with Timer() as t:
        for i,rec in enumerate(pool.imap(readTestRow, reader, chunksize=1000)): 
            data = np.vstack([data, rec])
            
            if not testMovies.has_key(rec[0]):
                 testMovies[rec[0]] = i
            
            if not testUsers.has_key(rec[1]):
                 testUsers[rec[1]] = i                
                 
            if i % 100000 == 0:
                print "{0} lines in {1}s ({2} lines/s)".format(i, t.elapsed, t.rate(i))
            
        

data = np.delete(data, 0, 0)
fp = np.memmap(outputTestfilename, dtype='int', mode='w+', shape=(len(data),2))
fp[:] = data[:]
fp.flush()
 
print "total {0} test records".format(i)           
print " total test {0} movies.".format(len(testMovies))
print " total test {0} users.".format(len(testUsers))

reader = csv.DictReader(open(inputTrainfilename))

data = np.empty([1,3],dtype='int')
trainUsers = {}
trainMovies = {}

with Timer() as t:
        for i,rec in enumerate(pool.imap(readTrainRow, reader, chunksize=1000)): 
            data = np.vstack([data, rec])
            
            if not trainMovies.has_key(rec[0]):
                 trainMovies[rec[0]] = i
            
            if not trainUsers.has_key(rec[1]):
                 trainUsers[rec[1]] = i                
                 
            if i % 100000 == 0:
                print "{0} lines in {1}s ({2} lines/s)".format(i, t.elapsed, t.rate(i))
            

data = np.delete(data, 0, 0)
fp = np.memmap(outputTrainfilename, dtype='int', mode='w+', shape=(len(data),3))
fp[:] = data[:]
fp.flush()
           
print " total train {0} movies.".format(len(trainMovies))
print " total train {0} users".format(len(trainUsers))
print "total {0} train records".format(i)






