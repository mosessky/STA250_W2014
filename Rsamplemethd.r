#!/usr/bin/Rscript

library(FastCSVSample)

# Start the clock!
ptm <- proc.time()

#get current working directory
initial.options <- commandArgs(trailingOnly = FALSE)
file.arg.name <- "--file="
script.name <- sub(file.arg.name, "", initial.options[grep(file.arg.name, initial.options)])
script.basename <- dirname(script.name)

#here you have to put in your own working directory
files<-list.files("/home/shikai/STA250/Homework1/data", pattern = "^[12].*.csv$", full.names = TRUE)

delay={}
x={}

for(file in files){
  
  #sample 2000 lines from each file
  tmp<-csvSample(file,2000)
  
  year<-as.numeric(sub(".*([0-9]{4})[^0-9]*\\.csv","\\1",file))
  #write(paste("year: ", year, sep=" "), stdout())
  if(year < 2008){
    
    x = as.integer(lapply(strsplit(tmp,","), function(y) y[[15]]))
    delay=c(delay,x)
  }else{
   
    x = as.integer(lapply(strsplit(tmp,","), function(y) y[[45]]))
    delay=c(delay,x)
  }
  
}

#mean 
avg = mean(delay, na.rm=TRUE)
varVal = var(delay, na.rm=TRUE)
medVal = median(delay, na.rm=TRUE)

#get the excution computation time
elapsed = proc.time()-ptm

write(paste("mean value: ", avg, sep=" "), stdout())
write(paste("variance: ",varVal, sep=" "), stdout())
write(paste("median value: ", medVal, sep=" "),stdout())
write(paste("elapsed time: ", elapsed['elapsed'], sep=" "),stdout())
write(paste("Computer Information: ", Sys.info()['machine'], sep=" "),stdout())