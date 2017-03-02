import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;
import Jama.*;


public class HW3 
{
    private static double[] train_x;  //training data of x which used to calculate m(x) s^2(x)
    private static double[] train_t;  //training data of t which used to calculate m(x) s^2(x)
    private static double[] read_train_xdata,read_train_tdata;  //array to store data read from file
    private static double[] test_x=new double[10];  //array to restore 10 new test point
    private static double[] test_t=new double[10];  //array to restore 10 true value of predicted value of new test point
    private static int num=0;   //number of lines in file
    private static int order;   //order , m in formula
    private static Matrix s;    //matrix S 
	private static final double alpha=0.005,beta=11.1;  
	private static double absolute_mean=0,average_relative_error=0;   
	private static double[] m_x=new double[10],s_square_x=new double[10],Norm_dis=new double[10];   //array to store result
	
	
	public static void main(String[] args) throws Exception
    {
		readTrainingData(args[0]);    //read training data
		readTestData(args[1]);        //read test data
		order=Integer.parseInt(args[2]);  //get order

		int test_size=num/10;   //because it's needed to run 10 times so divide training data into 10 groups
		train_t=new double[test_size];  
		train_x=new double[test_size];
		
		for(int cnt=0;cnt<10;cnt++)   //run 10 times
		{
			for(int test_cnt=0;test_cnt<test_size;test_cnt++)   //give values to training array
			{
				train_x[test_cnt]=read_train_xdata[cnt*test_size+test_cnt];
				train_t[test_cnt]=read_train_tdata[cnt*test_size+test_cnt];
			}
			m_x[cnt]=c_mx(cnt);  //get m(x)
			s_square_x[cnt]=c_s_square(cnt);  //get s^2(x)
			Norm_dis[cnt]=c_norm_dis(m_x[cnt],cnt);  //get probability of m(x)
			absolute_mean+=Math.abs(m_x[cnt]-test_t[cnt]);  //sum all absolute erroe
			average_relative_error+=Math.abs(m_x[cnt]-test_t[cnt])/test_t[cnt];  //sum all relative error
		}
		absolute_mean/=10;  //get absolute mean error
		average_relative_error/=10;  //get average relative error
		System.out.println("The absolute mean error is: "+absolute_mean);
		System.out.println("The average relative error is: "+average_relative_error);		
    }
	
	public static void readTrainingData(String s) throws Exception   // use scnner to read file
	{
		FileInputStream file = new FileInputStream(s); 
	    Scanner scanner = new Scanner(file);  
        while(scanner.hasNext())  
	    {
        	scanner.next();
	    	num++;
	    }
        file.close();
        scanner.close();
        read_train_xdata=new double[num];
        read_train_tdata=new double[num];
        file = new FileInputStream(s); 
	    scanner = new Scanner(file);  
	    int x_cnt=0,t_cnt=0;
        while(scanner.hasNext())  
	    {
        	String[] s_temp=scanner.next().split(",");   //split string
        	SimpleDateFormat df=new SimpleDateFormat("yyyy-MM-dd");   //define time pattern
        	Date d=df.parse(s_temp[0]);   //transform string into Date type
        	read_train_xdata[x_cnt++]=(double)(d.getTime()/86400000);  //change millisecond into day from UNIX time
        	read_train_tdata[t_cnt++]=Double.parseDouble(s_temp[1]);   // store price value
        }
        file.close();
        scanner.close();
	}
    
	public static void readTestData(String s) throws Exception   //use scanner to read file
	{
		FileInputStream file = new FileInputStream(s); 
		Scanner scanner = new Scanner(file); 
	    int cnt=0;
        while(cnt<10)  
	    {
	    	String[] s_temp=scanner.next().split(",");   //split string
	    	SimpleDateFormat df=new SimpleDateFormat("yyyy-MM-dd");  //define time pattern
        	Date d=df.parse(s_temp[0]);   //transform string to Date type
        	test_x[cnt]=(double)(d.getTime()/86400000);   //change millisecond into day from UNIX time 
        	test_t[cnt]=Double.parseDouble(s_temp[1]);  //store price value
        	cnt++;
        }
        file.close();
        scanner.close();
	}
	
	public static Matrix convert_phi(double x)   //generate phi(x) vector
	{
		Matrix temp=new Matrix(order+1,1,0);
		for(int cnt=0;cnt<order+1;cnt++)
			temp.set(cnt,0,Math.pow(x,cnt));		
		return temp;
	}
	
	public static Matrix c_s(int index)   //calculate matrix S
	{	
		s=new Matrix(order+1,order+1,0);
		final Matrix eye=Matrix.identity(order+1,order+1);
		for(int cnt=0;cnt<train_x.length;cnt++)
			s.plusEquals(convert_phi(train_x[cnt]).times(convert_phi(test_x[index]).transpose()));
		s.timesEquals(beta);
		s.plusEquals(eye.times(alpha));
		if(s.det()!=0)    //if S is not Singular
			return s.inverse();		
		else    //if S is Singular use svd to calculate its inverse
		{
			SingularValueDecomposition svd=s.svd();
			Matrix svd_u=svd.getU();
			Matrix svd_v=svd.getV();
			Matrix svd_s=svd.getS();
			for(int row=0;row<order+1;row++)
				for(int col=0;col<order+1;col++)
					if(svd_s.get(row,col)!=0)
						svd_s.set(row,col,1/svd_s.get(row,col));
			return svd_v.times(svd_s.transpose()).times(svd_u.transpose());		
		}	
	}
	
	public static double c_mx(int index)   //calculate m(x)
	{
		Matrix m_temp=new Matrix(order+1,1,0);
		for(int cnt=0;cnt<train_x.length;cnt++)
			m_temp.plusEquals(convert_phi(train_x[cnt]).times(train_t[cnt]));
		return convert_phi(test_x[index]).transpose().times(c_s(index)).times(m_temp).times(beta).det();
	}
	
	public static double c_s_square(int index)   //calculate s^2(x)
	{
		return convert_phi(test_x[index]).transpose().times(c_s(index)).times(convert_phi(test_x[index])).det()+1/beta;
	}

	public static double c_norm_dis(double t,int index)  //calculate probability with t value
	{
		return 1/(Math.sqrt(2*Math.PI*s_square_x[index]))*Math.exp(-1/(2*s_square_x[index])*(t-m_x[index])*(t-m_x[index]));
	}
}
 