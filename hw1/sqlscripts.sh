#! /bin/sh

#My own setting of MySQL
#please change 'password' to the one you used to access mysql database
MYSQL_ARGS="-u root -p'password' --local-infile tmpAviation"

#database name I used
DB="tmpAviation"
DELIM=","

#We assume we have everything (each year's aviation information) in csv file format
#Then, we write sql codes to import those csv files in the unzip folder of the aviation dataset
boldtable=false
bnewtable=false

for file in $(find $(pwd) -regex ".*\.\(csv\)") ; do

   year=$(echo $file | grep -o '[0-9]\{4\}')  
   if [ $year -lt 2008 ] 
   then
       table=oldtable
       if [ "$boldtable" = "false" ] 
       then
          boldtable=true
          FIELDS=$(head -1 "$file" | sed -e 's/'$DELIM'/` varchar(255),\n`/g' -e 's/\r//g')
          FIELDS='`'"$FIELDS"'` varchar(255)'    
          mysql $MYSQL_ARGS -e "
          DROP TABLE IF EXISTS $table;
          CREATE TABLE $table ($FIELDS);
	  "
       fi
   else
       table=newtable
       if [ "$bnewtable" = "false" ]
       then
          bnewtable=true
          FIELDS=$(head -1 "$file" | sed -e 's/'\"'//g' -e 's/,\+$//' -e 's/'$DELIM'/` varchar(255),\n`/g' -e 's/\r//g')
          FIELDS='`'"$FIELDS"'` varchar(255)'     
          mysql $MYSQL_ARGS -e "
          DROP TABLE IF EXISTS $table;
          CREATE TABLE $table ($FIELDS);
	  "
       fi
   fi
   mysql $MYSQL_ARGS -e "
   LOAD DATA INFILE '$file' INTO TABLE $table
   FIELDS TERMINATED BY '$DELIM'
   IGNORE 1 LINES
   ;
   "
done

mysql $MYSQL_ARGS -e "
   select AVG(delay) as avgVal, STD(delay) as stdVal from  ( select ArrDelay as delay from oldtable union all select ARR_DEL15 from newtable ) as tabledelay
       ;
       "





