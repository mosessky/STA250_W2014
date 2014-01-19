#! /usr/bin/env python

import os
import bz2
import sys

'''Please customized the following file and directory
info for your own data'''

#get path for this scripts
path=os.path.dirname(os.path.realpath(__file__))

#absolute file name
filename = path+"/Delays1987_2013.tar.bz2"

#set up connection for bzipfile input
con = bz2.BZ2File(filename,'r')
frqtable={}

#initial total number of observations
n = 0 

#initial start time
start = time.clock()

#input file line by line
for line in con:
      fieldList = line.split(",")
      #old file format input (find the field has ArrDelay)
      if "ArrDelay" in line:
        arrDelayind = fieldList.index("ArrDelay")
        binold = True
      #new file format input (find the field has ARR_DELAY)
      elif "ARR_DELAY" in line:
       arrDelayind = fieldList.index("\"ARR_DELAY\"")
       arrDelayind += 2
       binold = False
      #gather actual data once determine its input format
      else:
       if len(fieldList) > arrDelayind:
           if fieldList[arrDelayind] != 'NA' and fieldList[arrDelayind] != '':
		tmpind = float(fieldList[arrDelayind])
		#move calulation of mean and variance into later part
                '''
                #update mean and variance
		n += 1
                if n == 1:
                   avgVal = tmpind * 1.0
                   varVal = 0
                else:
                     varVal = (n-2.0)/(n-1.0) * varVal + (tmpind * 1.0  - avgVal)*(tmpind -avgVal) / n
                     avgVal +=  (tmpind - avgVal) / n
                '''
                #update frequency table
		if frqtable.has_key(tmpind):
                   frqtable[tmpind] = frqtable[tmpind] + 1
                else:
                   frqtable[tmpind] = 1
                

#close the connection
con.close()


#calculate mean and variance.
#Actually, we could complete this step in previous data gathering process
#but I think in frequency table, it would be faster
for key in frqtable.keys():
      
   if n == 0:
      #mean value
      avgVal = key
      #mean squared value
      sqVal = key * key
      #variance value
      varVal = 0
   else:
      #update  mean value
      avgVal = avgVal + 1.0 * frqtable[key] / (frqtable[key] + n) * (key -avgVal)  
      #update  mean squared value
      sqVal = sqVal + 1.0 * frqtable[key] / (frqtable[key] + n) * (key * key - sqVal)
      #update variance value
      varVal = sqVal - avgVal * avgVal
   
   n += frqtable[key]


#initial index   
index = 0
#median index
medind = int(n/2)

'''
loop dictionary to find median value:
current complexity O(l*l) (l is the length of the frequency dictionary)
If I write an efficient sort algorithm (O(l*log(l)) to sort the dictionationary by key, 
then the median value search could improve to O(l*log(l))
'''  
while (len(frqtable) > 0 and index < medind):
         minVal = 99999999.0
         #find minimum value in the dictionary
         for key in frqtable.keys():
            if key <= minVal:
                minVal = key
         #update index of minVal
         index += frqtable[minVal]
	 #if not reach median, then remove the item from dictionary
         #let the loop proceed to update index
         if index < medind:
              del frqtable[minVal]
         else:  
               medVal = minVal
               break
   
#get the excution computation time
elapsed = (time.clock() - start)

print "mean value: ", avgVal
print "variance: ", varVal
print "median value: ", medVal
print "excution time: ", elapsed
print "Computer Information: ", os.uname

