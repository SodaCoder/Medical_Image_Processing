import java.io.BufferedReader;
import java.io.*;
import java.util.*;  

public class javarun
{
	public static void main(String[] args) throws IOException
	{
		try
		{
			//ProcessBuilder builder = new ProcessBuilder("python", System.getProperty("user.dir") + "\\sample.py", "1", "4");
			Scanner sc= new Scanner(System.in); 
			System.out.print("Already Preprocessed? Y / N: ");  
			String choice = sc.nextLine();
			ProcessBuilder builder = new ProcessBuilder("python3", "./main.py", "../Test/Preprocessed_Dataset/C15SHARPCOVID-3446.png", choice);
			builder.redirectErrorStream(true);
			Process process = builder.start();
			BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
			BufferedReader readers = new BufferedReader(new InputStreamReader(process.getErrorStream()));
			String lines = null;
			while((lines = reader.readLine())!= null)
			{
				System.out.println("lines\n"+lines);
			}
			while((lines = readers.readLine())!= null)
			{
				System.out.println("error lines"+lines);
			}
			
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
	}
}
