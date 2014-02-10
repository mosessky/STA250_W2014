import java.io.*;
import java.util.concurrent.atomic.AtomicIntegerArray;
import java.util.concurrent.atomic.AtomicInteger;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Delays implements Runnable {


	String filename;
	//StringBuffer filename;
    private static  AtomicIntegerArray delays;  
    private static  AtomicInteger minVal = new AtomicInteger(-2250); 
    private static  AtomicInteger maxVal = new AtomicInteger(2250);
    
    public Delays(String name, AtomicIntegerArray delays ) {
	  this.filename = name;
	  this.delays = delays;
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
	    	this.delays.incrementAndGet(val - minVal.get());
		
	}
    }

    public void showTable() {
	for(int i = 0; i < delays.length() ; i++) {
	    if(delays.get(i) > 0)
		System.out.println( minVal.get() + i + ": " + delays.get(i));
	}
    }


}