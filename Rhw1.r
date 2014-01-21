#!/usr/bin/Rscript


# Start the clock!
ptm <- proc.time()

#get current working directory
initial.options <- commandArgs(trailingOnly = FALSE)
file.arg.name <- "--file="
script.name <- sub(file.arg.name, "", initial.options[grep(file.arg.name, initial.options)])
script.basename <- dirname(script.name)

#pipline data into sort function, then accumlate them to frequencies
con<-pipe(paste( "bzip2 -dc ", paste(sep="/", script.basename,"Delays1987_2013.tar.bz2") , "| cut -d',' -f15,45 |sed 's/^[^0-9.\\]\\+\\([0-9]*[\\\"\\,]\\)//g'|sort -n|uniq -c", sep = " "))
#con<-pipe("bzip2 -dc Homework.tar.bz2|cut -d',' -f15,45 |sed 's/^[^0-9.]\\+\\([0-9]*[\\\"\\,]\\)//g'|sed 's/\\.[0-9]*//g'|sort -n|uniq -c")

#Get the frequency list from pipe
frqList<-readLines(con)

#close connection
close(con)

#drop the first row, which contains number of NA values
#frqList<-frqList[3:(length(frqList)-2)]

#seperate result string into arrays with two numbers
frqTable <- read.table(textConnection(frqList),fill = TRUE)

#calculate mean and variance
n = 0
avg = 0
sqVal = 0
i = 1

while(i <= nrow(frqTable) ){
     #frequency of arrival delay
     frq = as.numeric(as.character(frqTable$V1[i]))
     if(is.na(frq)){
       frqTable = frqTable[-i,]  
       next
     }
     
     #actual arrival delay 
     numb = as.numeric(as.character(frqTable$V2[i]))
     if(is.na(numb)){
       frqTable = frqTable[-i,] 
       next
     }
       
     if(i == 1){
       #mean value
       avg = numb
       #mean squared value
       sqVal = numb * numb
       #variance value
       varVal = 0
     }else{
       #update  mean value
       avg = avg + 1.0 * frq / (frq * 1.0 + n) * (numb -avg)  
       #update  mean squared value
       sqVal = sqVal + 1.0 * frq / (frq * 1.0 + n) * (numb * numb - sqVal)
       #update variance value
       varVal = sqVal - avg * avg
     }
     
     n = n + frq
     i = i + 1
}


#initial index
index = 0
#median index
medind = as.integer(n/2)

for(i in 1:nrow(frqTable)){
  
  #frequency of arrival delay
  frq = as.numeric(as.character(frqTable$V1[i]))
  if(is.na(frq)){
    next
  }
  
  index = index + frq
  if(index >= medind){
    #store median value
    medVal = as.numeric(as.character(frqTable$V2[i]))
    break
  }
}

#get the excution computation time
elapsed = proc.time()-ptm

write(paste("mean value: ", avg, sep=" "), stdout())
write(paste("variance: ",varVal, sep=" "), stdout())
write(paste("median value: ", medVal, sep=" "),stdout())
write(paste("elapsed time: ", elapsed['elapsed'], sep=" "),stdout())
write(paste("Computer Information: ", Sys.info()['machine'], sep=" "),stdout())



