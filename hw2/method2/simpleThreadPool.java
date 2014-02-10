import java.io.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicInteger;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class simpleThreadPool{
	
	public static void main(String args[]) {

	    long tStart = System.currentTimeMillis();
	    File folder = new File(args[0]);
	    File[] listOfFiles = folder.listFiles();
	    AtomicIntegerArray delays  = new AtomicIntegerArray(4501);
	    AtomicInteger minVal = new AtomicInteger(-2250); 
	    AtomicInteger maxVal = new AtomicInteger(2250);
	    
		if(listOfFiles.length > 1) {
		    
			ExecutorService executor = Executors.newFixedThreadPool(8);
		    int i;

	        //initial multi-threads programming
		    for(i = 0; i < listOfFiles.length; i++) {
		      Delays worker = new Delays(listOfFiles[i].getAbsolutePath(), delays);
			  executor.execute(worker);
		    }

		    executor.shutdown();
		    while (!executor.isTerminated()) {
		    
		    }
		} else {
		    Delays d = new Delays(listOfFiles[0].getAbsolutePath(), delays);
		    d.run();
		}
		
		double meanVal = 0.0;
		double sdVal  = 0.0;
		double sqVal = 0.0;
	    double medianVal = 0.0;
		int count = 0;
		double tmpVal;
		for(int i =0; i <= 4500; i++){
			
			if(delays.get(i) == 0)
				continue;
			else{
				tmpVal = i+minVal.get();
				if(count == 0){
					meanVal = tmpVal;
					sqVal = tmpVal * tmpVal;
					sdVal = 0; 
				}else{
					meanVal = meanVal + 1.0 * delays.get(i)/(count + delays.get(i)) * (tmpVal - meanVal) ;
					sqVal = sqVal + 1.0 * delays.get(i)/(count + delays.get(i)) * (tmpVal * tmpVal - sqVal) ;
					sdVal = sqVal - meanVal * meanVal;
				}
				count += delays.get(i);
			}
		}
		sdVal = Math.sqrt(sdVal);
		int index = 0;
		for(int i =0; i <= 4500; i++){
			if(delays.get(i) == 0)
				continue;
			else{
				index += delays.get(i);
				if(index > count/2){
					medianVal = i + minVal.get();
					break;
				}
			}
		}
		
		long tEnd = System.currentTimeMillis();
		long tDelta = tEnd - tStart;
		double elapsedSeconds = tDelta / 1000.0;
		
		System.out.printf("Mean Value:  %f \n", meanVal);
		System.out.println("SD Value: "+ sdVal);
		System.out.println("Median Value: "+ medianVal);
		System.out.println("Elapsed Time: "+ elapsedSeconds);
		//Total number of processors or cores available to the JVM */
	    System.out.println("Available processors (cores): " + 
	        Runtime.getRuntime().availableProcessors());
	    /* Total memory currently available to the JVM */
	    System.out.println("Total memory available to JVM (bytes): " + 
	        Runtime.getRuntime().totalMemory());
		System.out.println("the program is ending");
	    }
   
}