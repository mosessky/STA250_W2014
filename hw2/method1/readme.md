Parallel Mean, SD, Median Value Computation
=================================================
In this homework, I write a simple java multithread program to compute mean, standard devivation and median value. 

-------------------------------------------------
The java program (Delays.java) reads the whole folder of *.csv airline delay data and computes the mean, median, and standard deviation. 

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
Meanwhile, I use a new laptop to run the program that's the reason why the 
computation time improves so much.I plan to test with threads pool services to 
achieve better computation time. 

