package MaxEntLDA;

import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import util.DocSentsCorpus;
import util.IOUtils;

public class MaxEntLDA {

	public static int OP = 0; // OPINION

	public static int AS = 1; // ASPECT

	public DocSentsCorpus trainData;

	//
	public int K = 20; // topic number

	public double alpha = 0.1;

	public double beta_a = 0.01;

	public double beta_o = 0.01;

	//
	public int niters = 1000;

	public int topN = 10;

	public List<List<Integer>> z_assign = new ArrayList<List<Integer>>();

	public List<List<List<Integer>>> y_assign = new ArrayList<List<List<Integer>>>();

	public void loadCorpus(String txtPath) {
		trainData = new DocSentsCorpus();
		trainData.loadCorpus(txtPath);
	}

	public int ndk[][];
	public int ndkSum[];
	public int nkw_a[][];
	public int nkwSum_a[];
	public int nkw_o[][];
	public int nkwSum_o[];

	public void initMaxEntLDA() {
		if (trainData == null)
			return;

		int M = trainData.M;
		int V = trainData.V;

		ndk = new int[M][K];
		ndkSum = new int[M];
		nkw_a = new int[K][V];
		nkwSum_a = new int[K];
		nkw_o = new int[K][V];
		nkwSum_o = new int[K];

		for (int m = 0; m != M; m++) {
			List<Integer> z_m = new ArrayList<Integer>();
			List<List<Integer>> y_m = new ArrayList<List<Integer>>();

			int S = trainData.doc_sent_w_id.get(m).size();

			for (int s = 0; s != S; s++) {
				int z_m_s = (int) Math.floor(Math.random() * K);

				List<Integer> y_m_s = new ArrayList<Integer>();

				int N = trainData.doc_sent_w_id.get(m).get(s).size();

				for (int n = 0; n != N; n++) {
					int w = trainData.doc_sent_w_id.get(m).get(s).get(n);

					double pro_o = trainData.doc_sent_w_pi.get(m).get(s).get(n);
					double pro_a = 1.0 - pro_o;

					int y_m_s_n = pro_o > pro_a ? OP : AS;

					if (y_m_s_n == OP) {
						nkw_o[z_m_s][w] += 1;
						nkwSum_o[z_m_s] += 1;
					}
					if (y_m_s_n == AS) {
						nkw_a[z_m_s][w] += 1;
						nkwSum_a[z_m_s] += 1;
					}

					y_m_s.add(y_m_s_n);

				}

				ndk[m][z_m_s] += 1;
				ndkSum[m] += 1;

				z_m.add(z_m_s);
				y_m.add(y_m_s);
			}

			z_assign.add(z_m);
			y_assign.add(y_m);
		}
	}

	public void sample_z(int m, int s) {
		int z_m_s = z_assign.get(m).get(s);

		int N = trainData.doc_sent_w_id.get(m).get(s).size();

		for (int n = 0; n != N; n++) {
			int w = trainData.doc_sent_w_id.get(m).get(s).get(n);
			int y_m_s_n = y_assign.get(m).get(s).get(n);
			if (y_m_s_n == OP) {
				nkw_o[z_m_s][w] -= 1;
				nkwSum_o[z_m_s] -= 1;
			}

			if (y_m_s_n == AS) {
				nkw_a[z_m_s][w] -= 1;
				nkwSum_a[z_m_s] -= 1;
			}
		}

		ndk[m][z_m_s] -= 1;
		ndkSum[m] -= 1;

		TIntIntMap wcnt_a = new TIntIntHashMap();
		TIntIntMap wcnt_o = new TIntIntHashMap();

		for (int n = 0; n != N; n++) {
			int w = trainData.doc_sent_w_id.get(m).get(s).get(n);
			int y_m_s_n = y_assign.get(m).get(s).get(n);
			if (y_m_s_n == OP) {
				wcnt_o.adjustOrPutValue(w, 1, 1);
			}
			if (y_m_s_n == AS) {
				wcnt_a.adjustOrPutValue(w, 1, 1);
			}
		}

		double Vbeta_o = trainData.V * beta_o;
		double Vbeta_a = trainData.V * beta_a;

		double[] pTable = new double[K];
		for (int k = 0; k != K; k++) {
			List<Double> devident_o = new ArrayList<Double>();
			List<Double> devident_a = new ArrayList<Double>();
			int oc = 0, ac = 0;
			for (int n = 0; n != N; n++) {
				if (y_assign.get(m).get(s).get(n) == OP) {
					devident_o.add(Vbeta_o + nkwSum_o[k] + oc);
					oc++;
				}
				if (y_assign.get(m).get(s).get(n) == AS) {
					devident_a.add(Vbeta_a + nkwSum_a[k] + ac);
					ac++;
				}
			}

			double expectTW = 1.0;
			int index_o = 0, index_a = 0;
			for (int w : wcnt_o.keys()) {
				int cnt = wcnt_o.get(w);
				for (int i = 0; i != cnt; i++) {
					expectTW *= (nkw_o[k][w] + beta_o + i)
							/ (devident_o.get(index_o));
					index_o++;
				}
			}
			for (int w : wcnt_a.keys()) {
				int cnt = wcnt_a.get(w);
				for (int i = 0; i != cnt; i++) {
					expectTW *= (nkw_a[k][w] + beta_a + i)
							/ (devident_a.get(index_a));
					index_a++;
				}
			}

			pTable[k] = (ndk[m][k] + alpha) / (ndkSum[m] + K * alpha)
					* expectTW;
		}

		for (int k = 1; k != K; k++) {
			pTable[k] += pTable[k - 1];
		}

		double r = Math.random() * pTable[K - 1];

		System.out.println("r=" + r + ";p[K]=" + pTable[K - 1]);

		for (int k = 0; k != K; k++) {
			if (pTable[k] > r) {
				z_m_s = k;
				break;
			}
		}

		for (int n = 0; n != N; n++) {
			int w = trainData.doc_sent_w_id.get(m).get(s).get(n);
			int y_m_s_n = y_assign.get(m).get(s).get(n);
			if (y_m_s_n == OP) {
				nkw_o[z_m_s][w] += 1;
				nkwSum_o[z_m_s] += 1;
			}

			if (y_m_s_n == AS) {
				nkw_a[z_m_s][w] += 1;
				nkwSum_a[z_m_s] += 1;
			}

		}

		ndk[m][z_m_s] += 1;
		ndkSum[m] += 1;

		z_assign.get(m).set(s, z_m_s);
	}

	public void sample_y_u(int m, int s, int n) {
		int z_m_s_n = z_assign.get(m).get(s);
		int y_m_s_n = y_assign.get(m).get(s).get(n);

		int w = trainData.doc_sent_w_id.get(m).get(s).get(n);
		if (y_m_s_n == OP) {
			nkw_o[z_m_s_n][w] -= 1;
			nkwSum_o[z_m_s_n] -= 1;
		}

		if (y_m_s_n == AS) {
			nkw_a[z_m_s_n][w] -= 1;
			nkwSum_a[z_m_s_n] -= 1;
		}

		double pro_o = trainData.doc_sent_w_pi.get(m).get(s).get(n);
		double pro_a = 1.0 - pro_o;

		double Vbeta_o = trainData.V * beta_o;
		double Vbeta_a = trainData.V * beta_a;

		double[] pTable = new double[2];
		pTable[0] = pro_o * (nkw_o[z_m_s_n][w] + beta_o)
				/ (nkwSum_o[z_m_s_n] + Vbeta_o);
		pTable[1] = pro_a * (nkw_a[z_m_s_n][w] + beta_a)
				/ (nkwSum_a[z_m_s_n] + Vbeta_a);

		for (int i = 1; i != 2; i++)
			pTable[i] += pTable[i - 1];

		double r = Math.random() * pTable[2 - 1];

		for (int i = 0; i != 2; i++) {
			if (pTable[i] > r) {
				switch (i) {
				case 0:
					y_m_s_n = OP;
					break;
				case 1:
					y_m_s_n = AS;
					break;
				}
				break;
			}
		}

		if (y_m_s_n == OP) {
			nkw_o[z_m_s_n][w] += 1;
			nkwSum_o[z_m_s_n] += 1;
		}

		if (y_m_s_n == AS) {
			nkw_a[z_m_s_n][w] += 1;
			nkwSum_a[z_m_s_n] += 1;
		}

		y_assign.get(m).get(s).set(n, y_m_s_n);
	}

	public void estimate() {
		for (int iter = 0; iter != niters; iter++) {

			System.out.println("Iteration: " + iter + " ...");

			for (int m = 0; m != trainData.doc_sent_w_id.size(); m++) {
				for (int s = 0; s != trainData.doc_sent_w_id.get(m).size(); s++) {
					sample_z(m, s);
				}
			}

			for (int m = 0; m != trainData.doc_sent_w_id.size(); m++) {
				for (int s = 0; s != trainData.doc_sent_w_id.get(m).size(); s++) {
					for (int n = 0; n != trainData.doc_sent_w_id.get(m).get(s)
							.size(); n++) {
						sample_y_u(m, s, n);
					}
				}
			}
		}
	}

	public double[][] compute_phi_o() {
		int V = trainData.V;
		double[][] phi = new double[K][V];
		for (int k = 0; k != K; k++)
			for (int w = 0; w != V; w++)
				phi[k][w] = (nkw_o[k][w] + beta_o) / (nkwSum_o[k] + V * beta_o);
		return phi;
	}

	public double[][] compute_phi_a() {
		int V = trainData.V;
		double[][] phi = new double[K][V];
		for (int k = 0; k != K; k++)
			for (int w = 0; w != V; w++)
				phi[k][w] = (nkw_a[k][w] + beta_a) / (nkwSum_a[k] + V * beta_a);
		return phi;
	}

	public double[][] term_score_a() {
		int V = trainData.V;
		double[][] tscore = new double[K][V];

		double[][] phi_a = this.compute_phi_a();
		for (int k = 0; k != K; k++) {
			for (int v = 0; v != V; v++) {
				double score = 1.0;
				for (int t = 0; t != K; t++) {
					score *= phi_a[t][v];
				}
				score = Math.log(phi_a[k][v] / Math.pow(score, 1.0 / K));
				score *= phi_a[k][v];
				tscore[k][v] = score;
			}
		}
		return tscore;
	}

	public double[][] term_score_o() {
		int V = trainData.V;
		double[][] tscore = new double[K][V];

		double[][] phi_o = this.compute_phi_o();
		for (int k = 0; k != K; k++) {
			for (int v = 0; v != V; v++) {
				double score = 1.0;
				for (int t = 0; t != K; t++) {
					score *= phi_o[t][v];
				}
				score = Math.log(phi_o[k][v] / Math.pow(score, 1.0 / K));
				score *= phi_o[k][v];
				tscore[k][v] = score;
			}
		}
		return tscore;
	}

	public ArrayList<List<Entry<String, Double>>> sorted_topicwords(
			double[] phi, int T) {
		ArrayList<List<Entry<String, Double>>> res = new ArrayList<List<Entry<String, Double>>>();
		HashMap<String, Double> term2weight = new HashMap<String, Double>();
		for (String term : trainData.word2id.keySet())
			term2weight.put(term, phi[trainData.word2id.get(term)]);

		List<Entry<String, Double>> pairs = new ArrayList<Entry<String, Double>>(
				term2weight.entrySet());
		Collections.sort(pairs, new Comparator<Entry<String, Double>>() {
			public int compare(Entry<String, Double> o1,
					Entry<String, Double> o2) {
				return (o2.getValue().compareTo(o1.getValue()));
			}
		});

		res.add(pairs);

		return res;
	}

	public ArrayList<List<Entry<String, Double>>> sorted_topicwords(
			double[][] phi, int T) {
		ArrayList<List<Entry<String, Double>>> res = new ArrayList<List<Entry<String, Double>>>();
		for (int k = 0; k != T; k++) {
			HashMap<String, Double> term2weight = new HashMap<String, Double>();
			for (String term : trainData.word2id.keySet())
				term2weight.put(term, phi[k][trainData.word2id.get(term)]);

			List<Entry<String, Double>> pairs = new ArrayList<Entry<String, Double>>(
					term2weight.entrySet());
			Collections.sort(pairs, new Comparator<Entry<String, Double>>() {
				public int compare(Entry<String, Double> o1,
						Entry<String, Double> o2) {
					return (o2.getValue().compareTo(o1.getValue()));
				}
			});

			res.add(pairs);
		}

		return res;
	}

	public void save_z_states(String outdir, String name) {
		BufferedWriter writer = IOUtils.getWriter(outdir + name, "utf-8");
		try {
			for (int m = 0; m != z_assign.size(); m++) {
				for (int s = 0; s != z_assign.get(m).size(); s++) {
					writer.append(z_assign.get(m).get(s) + " ");
				}
				writer.append("\n");
			}
			writer.flush();
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void print_topics(int top_n) {
		double phi_a[][] = this.compute_phi_a();
		double phi_o[][] = this.compute_phi_o();

		ArrayList<List<Entry<String, Double>>> pairs_list_a = this
				.sorted_topicwords(phi_a, K);
		ArrayList<List<Entry<String, Double>>> pairs_list_o = this
				.sorted_topicwords(phi_o, K);

		for (int k = 0; k != K; k++) {
			System.out.println("Topic: " + k + ":");
			System.out.println("Aspect:");
			for (int i = 0; i != top_n; i++) {
				System.out.println(pairs_list_a.get(k).get(i).getKey() + " "
						+ pairs_list_a.get(k).get(i).getValue());
			}

			System.out.println("\tOpinion:");
			for (int i = 0; i != top_n; i++) {
				System.out.println("\t" + pairs_list_o.get(k).get(i).getKey()
						+ " " + pairs_list_o.get(k).get(i).getValue());
			}
		}
	}

	// public void print_plain_topics(int top_n) {
	// double phi_a[][] = this.compute_phi_a();
	// double phi_o[][] = this.compute_phi_o();
	//
	//
	// ArrayList<List<Entry<String, Double>>> pairs_list_a = this
	// .sorted_topicwords(phi_a, K);
	// ArrayList<List<Entry<String, Double>>> pairs_list_o = this
	// .sorted_topicwords(phi_o, K);
	//
	// for (int k = 0; k != K; k++) {
	// for (int i = 0; i != top_n; i++) {
	// System.out.print(pairs_list_a.get(k).get(i).getKey()+" ");
	// }
	//
	// System.out.print("\t");
	// for (int i = 0; i != top_n; i++) {
	// System.out.print(pairs_list_o.get(k).get(i).getKey() + " ");
	// }
	// System.out.println();
	// }
	// }

	public void print_plain_topics(int top_n, String out) throws IOException {
		BufferedWriter writer = IOUtils.getWriter(out, "utf-8");

		double phi_a[][] = this.compute_phi_a();
		double phi_o[][] = this.compute_phi_o();

		ArrayList<List<Entry<String, Double>>> pairs_list_a = this
				.sorted_topicwords(phi_a, K);
		ArrayList<List<Entry<String, Double>>> pairs_list_o = this
				.sorted_topicwords(phi_o, K);

		for (int k = 0; k != K; k++) {
			for (int i = 0; i != top_n; i++) {
				writer.append(pairs_list_a.get(k).get(i).getKey() + " ");
			}

			writer.append("\t");
			for (int i = 0; i != top_n; i++) {
				writer.append(pairs_list_o.get(k).get(i).getKey() + " ");
			}
			writer.append("\n");
		}
		writer.close();
	}

	public void print_comp_topics(int top_n) {
		double phi_a[][] = this.compute_phi_a();
		double phi_o[][] = this.compute_phi_o();

		double tscore_a[][] = this.term_score_a();
		double tscore_o[][] = this.term_score_o();

		ArrayList<List<Entry<String, Double>>> pairs_list_sa = this
				.sorted_topicwords(tscore_a, K);
		ArrayList<List<Entry<String, Double>>> pairs_list_so = this
				.sorted_topicwords(tscore_o, K);

		ArrayList<List<Entry<String, Double>>> pairs_list_ta = this
				.sorted_topicwords(phi_a, K);
		ArrayList<List<Entry<String, Double>>> pairs_list_to = this
				.sorted_topicwords(phi_o, K);

		for (int k = 0; k != K; k++) {
			System.out.println("Topic: " + k + ":");
			System.out.println("Aspect:");
			for (int i = 0; i != top_n; i++) {
				System.out.println(pairs_list_ta.get(k).get(i).getKey() + " "
						+ pairs_list_sa.get(k).get(i).getKey());
			}

			System.out.println("\tOpinion:");
			for (int i = 0; i != top_n; i++) {
				System.out.println("\t" + pairs_list_to.get(k).get(i).getKey()
						+ " " + pairs_list_so.get(k).get(i).getKey());
			}
		}
	}

	public void print_topics_by_scores(int top_n) {
		double tscore_a[][] = this.term_score_a();
		double tscore_o[][] = this.term_score_o();

		ArrayList<List<Entry<String, Double>>> pairs_list_a = this
				.sorted_topicwords(tscore_a, K);
		ArrayList<List<Entry<String, Double>>> pairs_list_o = this
				.sorted_topicwords(tscore_o, K);

		for (int k = 0; k != K; k++) {
			System.out.println("Topic: " + k + ":");
			System.out.println("Aspect:");
			for (int i = 0; i != top_n; i++) {
				System.out.println(pairs_list_a.get(k).get(i).getKey() + " "
						+ pairs_list_a.get(k).get(i).getValue());
			}

			System.out.println("\tOpinion:");
			for (int i = 0; i != top_n; i++) {
				System.out.println("\t" + pairs_list_o.get(k).get(i).getKey()
						+ " " + pairs_list_o.get(k).get(i).getValue());
			}
		}
	}

	public void print_plain_topics_by_scores(int top_n) {
		double tscore_a[][] = this.term_score_a();
		double tscore_o[][] = this.term_score_o();

		ArrayList<List<Entry<String, Double>>> pairs_list_a = this
				.sorted_topicwords(tscore_a, K);
		ArrayList<List<Entry<String, Double>>> pairs_list_o = this
				.sorted_topicwords(tscore_o, K);

		for (int k = 0; k != K; k++) {
			for (int i = 0; i != top_n; i++) {
				System.out.print(pairs_list_a.get(k).get(i).getKey() + " ");
			}
			System.out.print("\t");
			for (int i = 0; i != top_n; i++) {
				System.out.print(pairs_list_o.get(k).get(i).getKey() + " ");
			}
			System.out.println();
		}
	}

	public static void main(String args[]) throws IOException {
		MaxEntLDA me_lda = new MaxEntLDA();
		me_lda.loadCorpus("/Users/zuoyuan/Desktop/experiment/APECWeibo.txt");
		me_lda.initMaxEntLDA();
		me_lda.estimate();
		// me_lda.save_z_states("C:/Users/zuoyuan/Desktop/ccMaxEnt-LDA/",
		// "states_sample3.txt");
		me_lda.print_topics(10);
		me_lda.print_plain_topics(10, "sample_output.txt");
		// me_lda.print_plain_topics_by_scores(20);
	}
}
