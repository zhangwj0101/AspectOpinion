package util;

import java.io.BufferedReader;
import java.io.IOException;

import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;

import gnu.trove.set.hash.TIntHashSet;
import gnu.trove.set.TIntSet;


public class ccDocSentsCorpus {

	public TIntObjectHashMap<String> id2word = new TIntObjectHashMap<String>();
	public TObjectIntHashMap<String> word2id = new TObjectIntHashMap<String>();
	
	public int C; // number of collections
	public int M;
	public int V;
	
	public int docsWids[][][];
	
	public double docsPis[][][];
	
	public int docsC[];
	
	public void readDocs(String filename) {
		System.out.println("Reading input from " + filename + " ...");
		
		BufferedReader reader = IOUtils.getReader(filename, "utf-8");
		M = 0;
		try {
			while(reader.readLine() != null)
				M ++;
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		docsWids = new int[M][][];
		docsPis = new double[M][][];
		
		docsC = new int[M];
		TIntSet cIdSet = new TIntHashSet();
		
		int docIndex = 0;
		String line = null;
		reader = IOUtils.getReader(filename, "utf-8");
		try {
			while ((line = reader.readLine()) != null) {
				String[] tabSplits = line.split("\t+");
				int S = tabSplits.length - 1;
				
				docsWids[docIndex] = new int[S][];
				docsPis[docIndex] = new double[S][];
				
				docsC[docIndex] = Integer.valueOf(tabSplits[0]);
				cIdSet.add(docsC[docIndex]);
				
				for (int s = 0; s != S; s++) {
					String[] spaceSplits = tabSplits[s+1].split(" +");
					int N = spaceSplits.length;
					
					docsWids[docIndex][s] = new int[N];
					docsPis[docIndex][s] = new double[N];
					
					for (int n = 0; n != N; n++) {
						String token = spaceSplits[n].substring(0, spaceSplits[n].indexOf(':'));
						if (!word2id.containsKey(token)) {
							word2id.put(token, word2id.size());
							id2word.put(id2word.size(), token);
						}
						String pro_o = spaceSplits[n].substring(spaceSplits[n].indexOf(':') + 1);
						docsWids[docIndex][s][n] = word2id.get(token);
						docsPis[docIndex][s][n] = Double.valueOf(pro_o);
					}
				}
				
				docIndex ++;
			}
			
			reader.close();
			
			C = cIdSet.size();
			V = word2id.size();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		System.out.println(C+" collections");
		System.out.println(M+" documents");
		System.out.println(V+" word types");
	}

}
