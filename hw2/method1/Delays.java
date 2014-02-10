import java.io.*;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicInteger;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Delays implements Runnable {


	String filename;
	//StringBuffer filename;
    static final AtomicIntegerArray delays = new AtomicIntegerArray(4501);
    static final AtomicInteger minVal = new AtomicInteger(-2250); 
    static final AtomicInteger maxVal = new AtomicInteger(2250);
    


    public Delays(String name) {
	  filename = name;
    	// filename = new StringBuffer(name);
    }


    public void run() {
        java.io.InputStream fstream;
	try {
	    fstream = new java.io.FileInputStream(filename.toString());
	    java.io.BufferedReader buf = new java.io.BufferedReader(new java.io.InputStreamReader(fstream));

	    buf.readLine(); // header
	    Pattern MY_PATTERN = Pattern.compile("\\d{4,}");
	    Matcher m = MY_PATTERN.matcher(filename.toString());
	    int year =0 ;
	    if (m.find())
	       year = Integer.parseInt(m.group(0));
	    
	    if(year < 2008)		
	        readRecords(buf, 14);
	    else
	    	readRecords(buf,44);
	    buf.close();
	} catch(Exception e) {
	    System.out.println("Problem processing " + filename.toString());
	    System.out.println(e.toString());
	    e.printStackTrace();
	}
    }


    public void readRecords(BufferedReader buf, int colNum) throws IOException {
	String line;
	int count = 0;
	while( (line = buf.readLine()) != null) {
	    String val = getDelay(line,colNum);
	    count ++;
	    storeDelay(val);
	}
	//We don't need to report this info in the final computation process
	System.out.println("Number of lines processed for " + filename.toString() + " " + count);
    }

    public String getDelay(String line, int colNum) {
	String[] els =  line.split(",");
	//	System.out.println(els[14]);
	return(els[colNum]);
    }

    protected void storeDelay(String value) {
	if(value != null && !value.isEmpty() && !value.equals("NA")) {  // not value != "NA" as in R!
	    int val = (int) Double.parseDouble(value);
	    if(val < minVal.get()  || val > maxVal.get()) 
		System.out.println("delay value problem " + val + ". Ignoring this value");
	    else
	    	delays.incrementAndGet(val - minVal.get());
		
	}
    }

    public void showTable() {
	for(int i = 0; i < delays.length() ; i++) {
	    if(delays.get(i) > 0)
		System.out.println( minVal.get() + i + ": " + delays.get(i));
	}
    }

    public static void main(String args[]) {

    long tStart = System.currentTimeMillis();
    File folder = new File(args[0]);
    File[] listOfFiles = folder.listFiles();
    
	if(listOfFiles.length > 1) {
	    Thread[] threads = new Thread[listOfFiles.length];
	    int i;

        //initial multi-threads programming
	    for(i = 0; i < listOfFiles.length; i++) {
		threads[i] = new Thread(new Delays(listOfFiles[i].getAbsolutePath()));
		threads[i].start();
		
	    }

	    for(i = 0; i < listOfFiles.length; i++) {
		try {
		    threads[i].join();
		} catch(InterruptedException e) {
		    System.out.println(e);
		}
	    }


	} else {
	    Delays d = new Delays(listOfFiles[0].getName());
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