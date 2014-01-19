#pipline data into sort function, then accumlate them to frequencies
con<-pipe("bzip2 -dc Delays1987_2012.tar.bz2| cut -d',' -f15|sort|uniq -c")

#Get the frequency list from pipe
frqList<-readLines(con)

#seperate result string into arrays with two numbers
frqTable <- read.table(textConnection(frqList))


#calculate mean
n = 0
avg = 0
for(i in 1:length(frqList) - 2){
     #frequency of arrival delay
     frq = as.numeric(frqTable$V1[i])
     #actual arrival delay 
     numb = as.numeric(frqTable$V2[i])
     
     if(avg == 0){
       avg = numb
     }else{
       avg = avg + (n + frq * numb /avg)/(n+frq)
     }
     
     n = n + frq
}

#calculate median and standard devision
for(i in 1:length(frqList)){
  
}
