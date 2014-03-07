Parallel Mean, SD, Median Value Computation
=================================================
In this homework, I tried three parallel methods to compute mean, standard devivation and median value. I put the codes into folders "method1", "method2" and "method3". 

-------------------------------------------------
The first method (Delays.java) reads the whole folder of *.csv airline delay data and computes the mean, median, and standard deviation. 

To run it, type commands: java Delays /home/XXX/Data 

Here, "/home/XXX/Data" is the absolute path that the data has been stored.

Results from Delays.class byte codes:
--------------------------------------------
Mean: 6.566436

SD Value: 31.553639122264

Median: 0.0

System Information:  (i7  2.4GHz), 8 Gb Memory
  
Computation Time: 1111.107 secs 

I notice four values are out of the data range (-2250, 2250).
That's the reason the result is slightly different from the previous homework.
Meanwhile, I use a new laptop to run the program , hence the computation time improves.I plan to test with threads pool services to achieve better computation time. 

When I use the first method, I notice that there was an oppurtunity to start a thread pool to complete the same work. Hence, I wrote a thread pool java application (simpleThreadPool.java). With manually twisting the total number of threads inside the thread pool, I get faster computation time.

Results from simpleThreadPool.class byte codes:
--------------------------------------------
Mean: 6.566436

SD Value: 31.553639122264

Median: 0.0

System Information:  (i7  2.4GHz), 8 Gb Memory
  
Computation Time: 855.07 secs


After doing these thread computations, I write a java hadoop program. Finally, I understand why we use bzip format for this dataset. The hadoop program (hadoop 2.2.0) could directly split the bzip file and work on it by multiple mappers. Another compressed format Lzo seems to be a preferred compressed data storage format. I didn't test the usage of different block sizes to increase mapper numbers. I am building a multiple node hadoop cluster by utilizing machines in our own lab.

Results from single node Hadoop:
--------------------------------------------
Mean: 6.5665

SD Value: 31.553639122264

Median: 0.0

System Information:  (i7  2.4GHz), 8 Gb Memory

Computation Time: 1380 secs
 

 
