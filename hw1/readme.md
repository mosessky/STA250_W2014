Single Process Mean, Variance and Median Value Calculation
==========================================================
In this homework, I wrote four simple single process scripts (simplescripts.py, sqlscripts.sh, Rhw1.r and Rsamplemethd.r) to 

find mean, variance and median value of flight departure time.

-----------------------------------------------------------------
The python script (simplescripts.py) directly reads from a bzip2 format file. 

To run it, please just put the script into the same foler as data (Delay1987_2013.tar.bz2).

Then, use shell command "./simplescripts.py". I hard code the file name, but it should be

just adjusted very easily. I make put detailed comments into the codes.


-------------------------------------------------------
The shell script (sqlscripts.sh) call MYSQL to build up a database named "tmpAviation". 

Inside this database, there are two tables "oldtable" and "newtable".

They store old format aviation information and new format aviation. 

I wrote this script to simulate the situation in case we want to save everything into 

database for research or any other query. I add in very light comments to make it readable.

To run it, please put the script into the same folder as unzipped text data folder.

Warning: the database loading process is very slow and also it requires lots of storage space 

(27G for data and several other gigabites for the database. So, if you don't have long storage 

or powerful computer, just escape this one.)


------------------------------------------------------
The R Script (Rhw1.r) is a very customized data processing methods compared with the previous 

two methods. It specifically calls shell commands to cut 15th and 45th column inside the text 

data. It's very efficient compared with the first two methods. The shell commands gather 

and sort all data. The acutall R codes did very few computations and runs in seconds, but the

called shell commands takes roughly 3000s seconds on my old computer.

To run it, please put the just put the script into the same foler as data (Delay1987_2013.tar.bz2).

Then, use shell command "./Rhw1.r". I hard code the file name, but it should be just adjusted very easily.


--------------------------------------
The another R Script (Rsamplemethd.r) adopts the FastCSVSample library provided by Dr.Duncan. It's fast

but not very accurate. Since I used very naive sample methods to get the records, so that might be the

reason that it is not very accurate.

To run this one, you have to open Rstudio and change the directory to all the txt files sit in.
 

*View the [source of the data.](http://eeyore.ucdavis.edu/stat250/Homeworks/hw1.html.)*

The python script extracts information in the column "ArrDelay" or "\"ARR_DELAY\"".

Then, we get the mean, variance and median.

*The script ignores the NA fields in the sample data.

Results from simplescripts.py:
------------------------------

Mean: 6.56650421703

Variance: 995.801720331

Median: 0.0

System Information: 'Linux', 'i686' (Intel Xeon(TM) CPU 3.00GHz), 

Memory 2.0 GB. (This is an old machine.)
  
Computation Time: 3691.43237805 secs

Results from Rhw1.r:
-------------------------------
Mean: 6.5665

Variance: 995.80172

Median: 0.0

System Information: 'Linux', 'i686' (Intel Xeon(TM) CPU 3.00GHz), 

Memory 2.0 GB. (This is an old machine.)
  
Computation Time: 2839.759 secs

Results from Rsamplemethd.r
--------------------------------
Mean: 7.23786

Variance: 707.256023

Median: 1

System Information: 'Linux', 'i686' (Intel Xeon(TM) CPU 3.00GHz), 

Memory 2.0 GB. (This is an old machine.)

Computation Time: 317.759 secs

Results from sqlscripts.sh
----------------------------
I only pass the test of this script on small dataset 10 years data 

(some from 1987-2007 and some from 2008-2012). For the full dataset,
my old machine runs out of memory.

