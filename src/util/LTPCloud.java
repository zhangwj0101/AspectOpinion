package util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.net.HttpURLConnection;
import java.net.MalformedURLException;
import java.net.URL;
import java.net.URLEncoder;
import java.util.ArrayList;
import java.util.List;

public class LTPCloud {

	public static String URL_BASE = "http://ltpapi.voicecloud.cn/analysis/?";
	
	public static String API_KEY = "43W6C138GKqJTSDWdbwNRXBzHvAbLEMQJWCLdBzQ";
	
	public static String FORMAT = "plain";
	
	public static String PATTERN = "pos";
	
	public static String getURL_BASE() {
		return URL_BASE;
	}

	public static void setURL_BASE(String uRL_BASE) {
		URL_BASE = uRL_BASE;
	}

	public static String getAPI_KEY() {
		return API_KEY;
	}

	public static void setAPI_KEY(String aPI_KEY) {
		API_KEY = aPI_KEY;
	}

	public static String getFORMAT() {
		return FORMAT;
	}

	public static void setFORMAT(String fORMAT) {
		FORMAT = fORMAT;
	}

	public static String getPATTERN() {
		return PATTERN;
	}

	public static void setPATTERN(String pATTERN) {
		PATTERN = pATTERN;
	}

	public static List<String> segmentation(String text) {
		List<String> segmentedSentences = new ArrayList<String>();
		
		try {
			text = URLEncoder.encode(text, "utf-8");

			String urlStr = URL_BASE + "api_key=" + API_KEY + "&text=" + text + "&format=" + FORMAT + "&pattern=" + PATTERN;
			
			URL request = new URL(urlStr);
			HttpURLConnection urlcon = (HttpURLConnection)request.openConnection();
			urlcon.connect();   
	        InputStream is = urlcon.getInputStream();
			
	        BufferedReader buffer = new BufferedReader(new InputStreamReader(is));
			
			String line = null;
			line = buffer.readLine();
			while(line != null) {
				segmentedSentences.add(line.trim());
				line = buffer.readLine();
			}
		} catch (UnsupportedEncodingException e) {
			e.printStackTrace();
		} catch (MalformedURLException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		return segmentedSentences;
	}
}
