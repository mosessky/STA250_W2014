/*
 Take each line of each CSV file and output the ArrDelay value and the constant 1
 to represent an observation for that particular delay value.
 We ignore the header and NA values.
 */

import java.io.IOException;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;

import org.apache.hadoop.mapreduce.Mapper;

public class ArrivalDelayMapper extends Mapper<LongWritable, Text, IntWritable, IntWritable> {

    public void map(LongWritable key, Text value, Context context) 
      	            throws IOException, InterruptedException 
    {
	int delay;
	String[] cols = value.toString().split(",");
	if(cols.length >= 14){
          if(!cols[0].equals("Year") || !cols[0].contains("Year"))
	  {
	      int year = 0;
	      try{
	         year = Integer.parseInt(cols[0]);
		}
		catch (RuntimeException e){
		   System.out.print("RuntimeException: ");
		   System.out.println(e.getMessage());
	           return;
	      }
              if(year < 2008){
	         if(!cols[14].equals("NA")){
		  delay = Integer.parseInt(cols[14]);
	          context.write(new IntWritable(delay), new IntWritable(1)); 
                 }	
              }	else{
               if(!cols[44].equals("NA")&&!(cols[44]==null)&&!(cols[44].trim().equals(""))){
		  delay = (int)(Double.parseDouble(cols[44]));
	          context.write(new IntWritable(delay), new IntWritable(1)); 
                 }	

              }
	  }
	}
    }

}
