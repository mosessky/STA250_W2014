import csv
from multiprocessing import Pool
import nltk
import sys
from itertools import groupby
from dateutil import parser as dateparser
import os



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


def processdata(filename):
    reader = csv.DictReader(open(filename))
    pool = Pool()
    categories = []
    texts = []
    
    with Timer() as t:
        for i,rec in enumerate(pool.imap(rowprocess, reader, chunksize=100)): 
           categories.append(rec[0])
           texts.append(rec[1])
           if i % 100000 == 0:
                print "{0} lines in {1}s ({2} lines/s)".format(i, t.elapsed, t.rate(i))
                if i == 500000:
                     return [categories, texts]        
    return [categories, texts]
def rowprocess(row):
    
    post_id = row['PostId']
    post_status = row['OpenStatus']
    title = row['Title']
    title += "\n"
    body = row['BodyMarkdown']
    
    tags = ""
    for i in range(1,6):
          tags = row["Tag%d"%i] + "," + tags
    tags += "\n"  
      
    """
    I can't make usage of the following characteristics (reputation, goodpost number and age) yet.
    One way to do it is after scipy creates the target sparse matrix, I add one column into that matrix.
    """
    reputation = row['ReputationAtPostCreation']
    goodposts = row['OwnerUndeletedAnswerCountAtPostTime']
    post_t = dateparser.parse(row['PostCreationDate'])
    user_t = dateparser.parse(row['OwnerCreationDate'])
    age = (post_t - user_t).total_seconds()
  
    symbols = ""
    
    lines = body.splitlines()
    code = []
    text = []
    for is_code, group in groupby(lines, lambda l: l.startswith('    ')):
         (code if is_code else text).append('\n'.join(group))
    for t in text:
        for sent in nltk.sent_tokenize(t):
             ss = sent.strip()
             if ss:
               if ss.endswith('?'):
                  symbols = "question ".join(symbols)
               if ss.endswith('!'):
                  symbols = "exclam ".join(symbols) 
               if ss.endswith('.'):
                  symbols = "period ".join(symbols)
               if ss.startswith('I '):
                  symbols = 'istart '.join(symbols) 
               if ss[0].isupper():
                  symbols = 'initcap '.join(symbols)
     
    symbols = symbols + "\n"
    
    return [post_status, title + body + symbols + tags ]    
                