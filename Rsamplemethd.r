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
files<-list.files("/home/shikai/STA250/Homework1", pattern = "^[12].*.csv$", full.names = TRUE)

delay={}
x={}

for(file in files){
  
  #sample 3000 lines from each file
  tmp<-csvSample(file,4000)
  
  year<-as.numeric(sub(".*([0-9]{4})[^0-9]*\\.csv","\\1",file))
  if(year<2008){
    tryCatch({x = as.integer(lapply(strsplit(tmp,","), function(x) x[[15]]))}, warning = function(x){x= NA})
    delay=c(delay,x)
  }else{
    tryCatch({x = as.integer(lapply(strsplit(tmp,","), function(x) x[[45]]))}, warning = function(x){x= NA})
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