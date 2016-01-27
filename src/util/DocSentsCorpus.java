package util;

import gnu.trove.map.hash.TIntObjectHashMap;
import gnu.trove.map.hash.TObjectIntHashMap;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class DocSentsCorpus {

	public TIntObjectHashMap<String> id2word = new TIntObjectHashMap<String>();
	public TObjectIntHashMap<String> word2id = new TObjectIntHashMap<String>();

	public List<List<List<Integer>>> doc_sent_w_id = new ArrayList<List<List<Integer>>>();

	public List<List<List<Double>>> doc_sent_w_pi = new ArrayList<List<List<Double>>>();

	public int M; // number of documents

	public int V; // size of vocabulary

	public String defaultCharset = "utf-8";

	/**
	 * document format: one sentence \t another sentence sentence format:
	 * token1:pi1 token2:pi2
	 */

	public void loadCorpus(String txtPath) {

		System.out.println("Loading training data from path: " + txtPath);

		BufferedReader txtReader = IOUtils.getReader(txtPath, defaultCharset);

		String lineDoc = null;

		try {

			lineDoc = txtReader.readLine();
			while (lineDoc != null) {
				
				if (lineDoc.compareTo("") == 0) {
					lineDoc = txtReader.readLine();
					continue;
				}

				List<List<Integer>> sent_w_id = new ArrayList<List<Integer>>();

				List<List<Double>> sent_w_pi = new ArrayList<List<Double>>();

				String[] sents = lineDoc.trim().split("\t+");

				for (String sent : sents) {

					List<Integer> w_id = new ArrayList<Integer>();

					List<Double> w_pi = new ArrayList<Double>();

					String[] token_pis = sent.trim().split(" +");

					for (String t_pi: token_pis) {
						String token = t_pi.substring(0, t_pi.indexOf(':')).toLowerCase();
						
						Double pi = Double.valueOf(t_pi.substring(t_pi
								.indexOf(':') + 1));

						if (!word2id.contains(token)) {
							word2id.put(token, word2id.size());
							id2word.put(word2id.get(token), token);
						}

						w_id.add(word2id.get(token));

						w_pi.add(pi);
					}

					sent_w_id.add(w_id);

					sent_w_pi.add(w_pi);

				}

				doc_sent_w_id.add(sent_w_id);

				doc_sent_w_pi.add(sent_w_pi);

				lineDoc = txtReader.readLine();
			}

			M = doc_sent_w_id.size();

			V = word2id.size();

			System.out.println("Load of the corpus is complete:");
			System.out.println("Training doc number: " + M);
			System.out.println("Vocabulary size: " + V);

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
