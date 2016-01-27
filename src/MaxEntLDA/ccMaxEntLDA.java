package MaxEntLDA;

import gnu.trove.list.TDoubleList;
import gnu.trove.list.array.TDoubleArrayList;
import gnu.trove.map.TIntIntMap;
import gnu.trove.map.hash.TIntIntHashMap;

import java.io.BufferedWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Random;
import java.util.Map.Entry;

import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.regression.LogisticRegression;
import jsat.regression.RegressionDataSet;

import util.IOUtils;
import util.MathUtil;
import util.ccDocSentsCorpus;

public class ccMaxEntLDA {

	public ccDocSentsCorpus trainData = null;

	public void loadCorpus(String txtFile) {
		trainData = new ccDocSentsCorpus();
		trainData.readDocs(txtFile);
	}

	public static double[] lables;
	public static int OP = 0;

	public static double[] coeffs;

	public static int AS = 1;

	public static int IN = 0;

	public static int SP = 1;

	public int KI = 5; // Independent topics shared by collections

	public int KS = 5; // Specific topics of each collection

	public double alpha = 0.1;

	public double beta_o = 0.01; // Beta for opinion distribution
	public double beta_a = 0.01; // Beta for aspect distribution

	public double gamma0 = 0.1;
	public double gamma1 = 0.1;

	public int niters = 1000;

	public int z_assigns[][];
	public int r_assigns[][];
	public int y_assigns[][][];

	public int ndk_i[][];
	public int ndkSum_i[];
	public int ndk_s[][];
	public int ndkSum_s[];
	public int nkw_ia[][];
	public int nkwSum_ia[];
	public int nkw_io[][][];
	public int nkwSum_io[][];
	public int nkw_so[][][];
	public int nkwSum_so[][];
	public int nkw_sa[][][];
	public int nkwSum_sa[][];
	public int ndr[][];
	public int ndrSum[];

	public void initialize() {
		System.out.println("Initializing...");
		int M = trainData.M;
		int V = trainData.V;
		int C = trainData.C;

		// int K = KI + KS;

		z_assigns = new int[M][];
		r_assigns = new int[M][];
		y_assigns = new int[M][][];

		ndk_i = new int[M][KI];
		ndkSum_i = new int[M];
		ndk_s = new int[M][KS];
		ndkSum_s = new int[M];

		ndr = new int[M][2];
		ndrSum = new int[M];

		nkw_ia = new int[KI][V];
		nkwSum_ia = new int[KI];
		nkw_io = new int[C][KI][V];
		nkwSum_io = new int[C][KI];
		nkw_so = new int[C][KS][V];
		nkwSum_so = new int[C][KS];
		nkw_sa = new int[C][KS][V];
		nkwSum_sa = new int[C][KS];

		coeffs = new double[KS + KI];
		Arrays.fill(coeffs, 1.0);

		Random rand = new Random(System.currentTimeMillis());

		for (int m = 0; m != M; m++) {
			int S = trainData.docsWids[m].length;

			int c = trainData.docsC[m];

			z_assigns[m] = new int[S];
			r_assigns[m] = new int[S];
			y_assigns[m] = new int[S][];

			for (int s = 0; s != S; s++) {
				int z = 0;
				int r = rand.nextInt(2);

				// update counts
				ndr[m][r] += 1;
				ndrSum[m] += 1;
				if (r == IN) {
					z = rand.nextInt(KI);
					ndk_i[m][z] += 1;
					ndkSum_i[m] += 1;
				}

				if (r == SP) {
					z = rand.nextInt(KS);
					ndk_s[m][z] += 1;
					ndkSum_s[m] += 1;
				}

				// record assignments
				z_assigns[m][s] = z;
				r_assigns[m][s] = r;

				int N = trainData.docsWids[m][s].length;

				y_assigns[m][s] = new int[N];
				for (int n = 0; n != N; n++) {
					int w = trainData.docsWids[m][s][n];
					double pro_o = trainData.docsPis[m][s][n];
					double pro_a = 1.0 - pro_o;

					int y = pro_o > pro_a ? OP : AS;
					// record assignments
					y_assigns[m][s][n] = y;

					// update counts
					if (y == OP && r == IN) {
						nkw_io[c][z][w] += 1;
						nkwSum_io[c][z] += 1;
					}

					if (y == OP && r == SP) {
						nkw_so[c][z][w] += 1;
						nkwSum_so[c][z] += 1;
					}

					if (y == AS && r == IN) {
						nkw_ia[z][w] += 1;
						nkwSum_ia[z] += 1;
					}

					if (y == AS && r == SP) {
						nkw_sa[c][z][w] += 1;
						nkwSum_sa[c][z] += 1;
					}
				}
			}
		}
	}

	public void sample_z_r(int m, int s) {
		int c = trainData.docsC[m];
		int z = z_assigns[m][s];
		int r = r_assigns[m][s];

		int N = trainData.docsWids[m][s].length;

		if (r == IN) {
			ndk_i[m][z] -= 1;
			ndkSum_i[m] -= 1;
		}
		if (r == SP) {
			ndk_s[m][z] -= 1;
			ndkSum_s[m] -= 1;
		}

		ndr[m][r] -= 1;
		ndrSum[m] -= 1;

		for (int n = 0; n != N; n++) {
			int w = trainData.docsWids[m][s][n];
			int y = y_assigns[m][s][n];
			if (y == OP && r == IN) {
				nkw_io[c][z][w] -= 1;
				nkwSum_io[c][z] -= 1;
			}

			if (y == OP && r == SP) {
				nkw_so[c][z][w] -= 1;
				nkwSum_so[c][z] -= 1;
			}

			if (y == AS && r == IN) {
				nkw_ia[z][w] -= 1;
				nkwSum_ia[z] -= 1;
			}

			if (y == AS && r == SP) {
				nkw_sa[c][z][w] -= 1;
				nkwSum_sa[c][z] -= 1;
			}
		}

		TIntIntMap wcnt_a = new TIntIntHashMap();
		TIntIntMap wcnt_o = new TIntIntHashMap();

		for (int n = 0; n != N; n++) {
			int w = trainData.docsWids[m][s][n];
			int y = y_assigns[m][s][n];
			if (y == OP) {
				wcnt_o.adjustOrPutValue(w, 1, 1);
			}
			if (y == AS) {
				wcnt_a.adjustOrPutValue(w, 1, 1);
			}
		}

		double Vbeta_o = trainData.V * beta_o;
		double Vbeta_a = trainData.V * beta_a;

		double[] pTable = new double[KI + KS];
		int k = 0;
		for (; k != KI; k++) {
			TDoubleList devident_o = new TDoubleArrayList();
			TDoubleList devident_a = new TDoubleArrayList();
			int oc = 0, ac = 0;
			for (int n = 0; n != N; n++) {
				if (y_assigns[m][s][n] == OP) {
					devident_o.add(Vbeta_o + nkwSum_io[c][k] + oc);
					oc++;
				}
				if (y_assigns[m][s][n] == AS) {
					devident_a.add(Vbeta_a + nkwSum_ia[k] + ac);
					ac++;
				}
			}

			double expectTW = 1.0;
			int index_o = 0, index_a = 0;
			for (int w : wcnt_o.keys()) {
				int cnt = wcnt_o.get(w);
				for (int i = 0; i != cnt; i++) {
					expectTW *= (nkw_io[c][k][w] + beta_o + i)
							/ (devident_o.get(index_o));
					index_o++;
				}
			}
			for (int w : wcnt_a.keys()) {
				int cnt = wcnt_a.get(w);
				for (int i = 0; i != cnt; i++) {
					expectTW *= (nkw_ia[k][w] + beta_a + i)
							/ (devident_a.get(index_a));
					index_a++;
				}
			}
			
			// change
			pTable[k] = updateParmas(m) * (ndr[m][IN] + gamma0)
					/ (ndrSum[m] + gamma0 + gamma1) * (ndk_i[m][k] + alpha)
					/ (ndkSum_i[m] + KI * alpha) * expectTW;
		}

		for (; k != KI + KS; k++) {
			TDoubleList devident_o = new TDoubleArrayList();
			TDoubleList devident_a = new TDoubleArrayList();
			int oc = 0, ac = 0;
			for (int n = 0; n != N; n++) {
				if (y_assigns[m][s][n] == OP) {
					devident_o.add(Vbeta_o + nkwSum_so[c][k - KI] + oc);
					oc++;
				}
				if (y_assigns[m][s][n] == AS) {
					devident_a.add(Vbeta_a + nkwSum_sa[c][k - KI] + ac);
					ac++;
				}
			}

			double expectTW = 1.0;
			int index_o = 0, index_a = 0;
			for (int w : wcnt_o.keys()) {
				int cnt = wcnt_o.get(w);
				for (int i = 0; i != cnt; i++) {
					expectTW *= (nkw_so[c][k - KI][w] + beta_o + i)
							/ (devident_o.get(index_o));
					index_o++;
				}
			}
			for (int w : wcnt_a.keys()) {
				int cnt = wcnt_a.get(w);
				for (int i = 0; i != cnt; i++) {
					expectTW *= (nkw_sa[c][k - KI][w] + beta_a + i)
							/ (devident_a.get(index_a));
					index_a++;
				}
			}

			// change
			pTable[k] = updateParmas(m) * (ndr[m][SP] + gamma1)
					/ (ndrSum[m] + gamma0 + gamma1)
					* (ndk_s[m][k - KI] + alpha) / (ndkSum_s[m] + KS * alpha)
					* expectTW;
		}

		for (int i = 1; i != KI + KS; i++) {
			pTable[i] += pTable[i - 1];
		}

		double u = Math.random() * pTable[KI + KS - 1];

		for (int i = 0; i != KI + KS; i++) {
			if (pTable[i] > u) {
				if (i < KI) {
					r = IN;
					z = i;
				} else {
					r = SP;
					z = i - KI;
				}
				break;
			}
		}

		if (r == IN) {
			ndk_i[m][z] += 1;
			ndkSum_i[m] += 1;
		}
		if (r == SP) {
			ndk_s[m][z] += 1;
			ndkSum_s[m] += 1;
		}

		ndr[m][r] += 1;
		ndrSum[m] += 1;

		for (int n = 0; n != N; n++) {
			int w = trainData.docsWids[m][s][n];
			int y = y_assigns[m][s][n];
			if (y == OP && r == IN) {
				nkw_io[c][z][w] += 1;
				nkwSum_io[c][z] += 1;
			}

			if (y == OP && r == SP) {
				nkw_so[c][z][w] += 1;
				nkwSum_so[c][z] += 1;
			}

			if (y == AS && r == IN) {
				nkw_ia[z][w] += 1;
				nkwSum_ia[z] += 1;
			}

			if (y == AS && r == SP) {
				nkw_sa[c][z][w] += 1;
				nkwSum_sa[c][z] += 1;
			}
		}

		z_assigns[m][s] = z;
		r_assigns[m][s] = r;

	}

	public void sample_y(int m, int s, int n) {
		int c = trainData.docsC[m];
		int z = z_assigns[m][s];
		int r = r_assigns[m][s];
		int y = y_assigns[m][s][n];

		int w = trainData.docsWids[m][s][n];
		if (y == OP && r == IN) {
			nkw_io[c][z][w] -= 1;
			nkwSum_io[c][z] -= 1;
		}

		if (y == OP && r == SP) {
			nkw_so[c][z][w] -= 1;
			nkwSum_so[c][z] -= 1;
		}

		if (y == AS && r == IN) {
			nkw_ia[z][w] -= 1;
			nkwSum_ia[z] -= 1;
		}

		if (y == AS && r == SP) {
			nkw_sa[c][z][w] -= 1;
			nkwSum_sa[c][z] -= 1;
		}

		double pro_o = trainData.docsPis[m][s][n];
		double pro_a = 1.0 - pro_o;

		double Vbeta_o = trainData.V * beta_o;
		double Vbeta_a = trainData.V * beta_a;

		double[] pTable = new double[2];
		if (r == IN) {
			pTable[0] = pro_o * (nkw_io[c][z][w] + beta_o)
					/ (nkwSum_io[c][z] + Vbeta_o);
			pTable[1] = pro_a * (nkw_ia[z][w] + beta_a)
					/ (nkwSum_ia[z] + Vbeta_a);
		}

		if (r == SP) {
			pTable[0] = pro_o * (nkw_so[c][z][w] + beta_o)
					/ (nkwSum_so[c][z] + Vbeta_o);
			pTable[1] = pro_a * (nkw_sa[c][z][w] + beta_a)
					/ (nkwSum_sa[c][z] + Vbeta_a);
		}

		for (int i = 1; i != 2; i++)
			pTable[i] += pTable[i - 1];

		double u = Math.random() * pTable[2 - 1];

		for (int i = 0; i != 2; i++) {
			if (pTable[i] > u) {
				switch (i) {
				case 0:
					y = OP;
					break;
				case 1:
					y = AS;
					break;
				}
				break;
			}
		}

		if (y == OP && r == IN) {
			nkw_io[c][z][w] += 1;
			nkwSum_io[c][z] += 1;
		}

		if (y == OP && r == SP) {
			nkw_so[c][z][w] += 1;
			nkwSum_so[c][z] += 1;
		}

		if (y == AS && r == IN) {
			nkw_ia[z][w] += 1;
			nkwSum_ia[z] += 1;
		}

		if (y == AS && r == SP) {
			nkw_sa[c][z][w] += 1;
			nkwSum_sa[c][z] += 1;
		}

		y_assigns[m][s][n] = y;
	}

	public void estimate() {

		for (int i = 0; i != niters; i++) {
			System.out.println("Iteration " + i + " ...");
			for (int m = 0; m != trainData.M; m++) {
				for (int s = 0; s != trainData.docsWids[m].length; s++) {
					sample_z_r(m, s);
				}
			}
			getCoefficents();
			for (int m = 0; m != trainData.M; m++) {
				for (int s = 0; s != trainData.docsWids[m].length; s++) {
					for (int n = 0; n != trainData.docsWids[m][s].length; n++) {
						sample_y(m, s, n);
					}
				}
			}
		}
	}

	public double[][][] compute_phi_so() {
		int V = trainData.V;
		int C = trainData.C;

		double[][][] phi = new double[C][KS][V];
		for (int c = 0; c != C; c++)
			for (int k = 0; k != KS; k++)
				for (int w = 0; w != V; w++)
					phi[c][k][w] = (nkw_so[c][k][w] + beta_o)
							/ (nkwSum_so[c][k] + V * beta_o);
		return phi;
	}

	public double[][][] compute_phi_sa() {
		int V = trainData.V;
		int C = trainData.C;

		double[][][] phi = new double[C][KS][V];
		for (int c = 0; c != C; c++)
			for (int k = 0; k != KS; k++)
				for (int w = 0; w != V; w++)
					phi[c][k][w] = (nkw_sa[c][k][w] + beta_a)
							/ (nkwSum_sa[c][k] + V * beta_a);
		return phi;
	}

	public double[][] compute_phi_ia() {
		int V = trainData.V;

		double[][] phi = new double[KI][V];
		for (int k = 0; k != KI; k++)
			for (int w = 0; w != V; w++)
				phi[k][w] = (nkw_ia[k][w] + beta_a)
						/ (nkwSum_ia[k] + V * beta_a);
		return phi;
	}

	public double[][][] compute_phi_io() {
		int V = trainData.V;
		int C = trainData.C;

		double[][][] phi = new double[C][KI][V];
		for (int c = 0; c != C; c++)
			for (int k = 0; k != KI; k++)
				for (int w = 0; w != V; w++)
					phi[c][k][w] = (nkw_io[c][k][w] + beta_o)
							/ (nkwSum_io[c][k] + V * beta_o);
		return phi;
	}

	public double[][] term_score_ia() {
		int V = trainData.V;
		double[][] tscore = new double[KI][V];

		double[][] phi_a = this.compute_phi_ia();
		for (int k = 0; k != KI; k++) {
			for (int v = 0; v != V; v++) {
				double score = 1.0;
				for (int t = 0; t != KI; t++) {
					score *= phi_a[t][v];
				}
				score = Math.log(phi_a[k][v] / Math.pow(score, 1.0 / KI));
				score *= phi_a[k][v];
				tscore[k][v] = score;
			}
		}
		return tscore;
	}

	public double[][][] term_score_io() {
		int C = trainData.C;
		int V = trainData.V;
		double[][][] tscore = new double[C][KI][V];

		double[][][] phi_o = this.compute_phi_io();
		for (int c = 0; c != C; c++)
			for (int k = 0; k != KI; k++) {
				for (int v = 0; v != V; v++) {
					double score = 1.0;
					for (int t = 0; t != KI; t++) {
						score *= phi_o[c][t][v];
					}
					score = Math
							.log(phi_o[c][k][v] / Math.pow(score, 1.0 / KI));
					score *= phi_o[c][k][v];
					tscore[c][k][v] = score;
				}
			}
		return tscore;
	}

	public double[][][] term_score_sa() {
		int C = trainData.C;
		int V = trainData.V;
		double[][][] tscore = new double[C][KS][V];

		double[][][] phi_a = this.compute_phi_sa();
		for (int c = 0; c != C; c++)
			for (int k = 0; k != KS; k++) {
				for (int v = 0; v != V; v++) {
					double score = 1.0;
					for (int t = 0; t != KS; t++) {
						score *= phi_a[c][t][v];
					}
					score = Math
							.log(phi_a[c][k][v] / Math.pow(score, 1.0 / KS));
					score *= phi_a[c][k][v];
					tscore[c][k][v] = score;
				}
			}
		return tscore;
	}

	public double[][][] term_score_so() {
		int C = trainData.C;
		int V = trainData.V;
		double[][][] tscore = new double[C][KS][V];

		double[][][] phi_o = this.compute_phi_so();
		for (int c = 0; c != C; c++)
			for (int k = 0; k != KS; k++) {
				for (int v = 0; v != V; v++) {
					double score = 1.0;
					for (int t = 0; t != KS; t++) {
						score *= phi_o[c][t][v];
					}
					score = Math
							.log(phi_o[c][k][v] / Math.pow(score, 1.0 / KS));
					score *= phi_o[c][k][v];
					tscore[c][k][v] = score;
				}
			}
		return tscore;
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

	public void printOneTopic(List<List<Entry<String, Double>>> pairs_list,
			String name, int k, int topN) {
		System.out.println(name + " " + k + ":");
		for (int i = 0; i != topN; i++) {
			System.out.println(pairs_list.get(k).get(i).getKey() + " "
					+ pairs_list.get(k).get(i).getValue());
		}
	}

	public String printPlainOneTopic(
			List<List<Entry<String, Double>>> pairs_list, String end, int k,
			int topN) {
		StringBuilder sbuilder = new StringBuilder("");
		for (int i = 0; i != topN; i++) {
			sbuilder.append(pairs_list.get(k).get(i).getKey() + " ");
		}
		sbuilder.append(end);
		return sbuilder.toString();
	}

	public void printAllTopics(int topN) {
		double phi_ia[][] = this.compute_phi_ia();
		double phi_io[][][] = this.compute_phi_io();

		List<List<Entry<String, Double>>> pairs_list_ia = this
				.sorted_topicwords(phi_ia, KI);
		List<List<List<Entry<String, Double>>>> pairs_lists_io = new ArrayList<List<List<Entry<String, Double>>>>();
		for (int c = 0; c != trainData.C; c++) {
			pairs_lists_io.add(sorted_topicwords(phi_io[c], KI));
		}

		for (int k = 0; k != KI; k++) {
			printOneTopic(pairs_list_ia, "Independent aspect", k, topN);
			for (int c = 0; c != trainData.C; c++) {
				printOneTopic(pairs_lists_io.get(c), "Collection " + c
						+ " opnion", k, topN);
			}
		}

		double phi_sa[][][] = this.compute_phi_sa();
		double phi_so[][][] = this.compute_phi_so();
		List<List<List<Entry<String, Double>>>> pairs_lists_so = new ArrayList<List<List<Entry<String, Double>>>>();
		for (int c = 0; c != trainData.C; c++) {
			pairs_lists_so.add(sorted_topicwords(phi_so[c], KS));
		}
		List<List<List<Entry<String, Double>>>> pairs_lists_sa = new ArrayList<List<List<Entry<String, Double>>>>();
		for (int c = 0; c != trainData.C; c++) {
			pairs_lists_sa.add(sorted_topicwords(phi_sa[c], KS));
		}

		for (int c = 0; c != trainData.C; c++) {
			for (int k = 0; k != KS; k++) {
				printOneTopic(pairs_lists_sa.get(c), "Collection " + c
						+ " Specific aspect", k, topN);
				printOneTopic(pairs_lists_so.get(c), "Collection " + c
						+ " Specific opnion", k, topN);
			}
		}
	}

	// public void printPlainAllTopics(int topN) {
	// double phi_ia[][] = this.compute_phi_ia();
	// double phi_io[][][] = this.compute_phi_io();
	//
	// List<List<Entry<String, Double>>> pairs_list_ia = this
	// .sorted_topicwords(phi_ia, KI);
	// List<List<List<Entry<String, Double>>>> pairs_lists_io = new
	// ArrayList<List<List<Entry<String, Double>>>>();
	// for (int c = 0; c != trainData.C; c++) {
	// pairs_lists_io.add(sorted_topicwords(phi_io[c], KI));
	// }
	//
	// for (int k = 0; k != KI; k++) {
	// for (int c = 0; c != trainData.C; c++) {
	// printPlainOneTopic(pairs_list_ia, "\t", k, topN);
	// printPlainOneTopic(pairs_lists_io.get(c), "\n", k, topN);
	// }
	// }
	//
	// double phi_sa[][][] = this.compute_phi_sa();
	// double phi_so[][][] = this.compute_phi_so();
	// List<List<List<Entry<String, Double>>>> pairs_lists_so = new
	// ArrayList<List<List<Entry<String, Double>>>>();
	// for (int c = 0; c != trainData.C; c++) {
	// pairs_lists_so.add(sorted_topicwords(phi_so[c], KS));
	// }
	// List<List<List<Entry<String, Double>>>> pairs_lists_sa = new
	// ArrayList<List<List<Entry<String, Double>>>>();
	// for (int c = 0; c != trainData.C; c++) {
	// pairs_lists_sa.add(sorted_topicwords(phi_sa[c], KS));
	// }
	//
	// for (int c = 0; c != trainData.C; c++) {
	// for (int k = 0; k != KS; k++) {
	// printPlainOneTopic(pairs_lists_sa.get(c), "\t", k, topN);
	// printPlainOneTopic(pairs_lists_so.get(c), "\n", k, topN);
	// }
	// }
	// }

	public void printPlainAllTopics(int topN, String out1, String out2)
			throws IOException {
		BufferedWriter writer1 = IOUtils.getWriter(out1, "utf-8");
		BufferedWriter writer2 = IOUtils.getWriter(out2, "utf-8");

		double phi_ia[][] = this.compute_phi_ia();
		double phi_io[][][] = this.compute_phi_io();

		List<List<Entry<String, Double>>> pairs_list_ia = this
				.sorted_topicwords(phi_ia, KI);
		List<List<List<Entry<String, Double>>>> pairs_lists_io = new ArrayList<List<List<Entry<String, Double>>>>();
		for (int c = 0; c != trainData.C; c++) {
			pairs_lists_io.add(sorted_topicwords(phi_io[c], KI));
		}

		double phi_sa[][][] = this.compute_phi_sa();
		double phi_so[][][] = this.compute_phi_so();
		List<List<List<Entry<String, Double>>>> pairs_lists_so = new ArrayList<List<List<Entry<String, Double>>>>();
		for (int c = 0; c != trainData.C; c++) {
			pairs_lists_so.add(sorted_topicwords(phi_so[c], KS));
		}
		List<List<List<Entry<String, Double>>>> pairs_lists_sa = new ArrayList<List<List<Entry<String, Double>>>>();
		for (int c = 0; c != trainData.C; c++) {
			pairs_lists_sa.add(sorted_topicwords(phi_sa[c], KS));
		}

		BufferedWriter writer = null;
		for (int c = 0; c != trainData.C; c++) {
			if (c == 0)
				writer = writer1;
			else
				writer = writer2;
			for (int k = 0; k != KI; k++) {
				writer.append(printPlainOneTopic(pairs_list_ia, "\t", k, topN));
				writer.append(printPlainOneTopic(pairs_lists_io.get(c), "\n",
						k, topN));
			}
			for (int k = 0; k != KS; k++) {
				writer.append(printPlainOneTopic(pairs_lists_sa.get(c), "\t",
						k, topN));
				writer.append(printPlainOneTopic(pairs_lists_so.get(c), "\n",
						k, topN));
			}
		}
		writer1.close();
		writer2.close();
	}

	public void printAllTopics_tscore(int topN) {
		double tscore_ia[][] = this.term_score_ia();
		double tscore_io[][][] = this.term_score_io();

		List<List<Entry<String, Double>>> pairs_list_ia = this
				.sorted_topicwords(tscore_ia, KI);
		List<List<List<Entry<String, Double>>>> pairs_lists_io = new ArrayList<List<List<Entry<String, Double>>>>();
		for (int c = 0; c != trainData.C; c++) {
			pairs_lists_io.add(sorted_topicwords(tscore_io[c], KI));
		}

		for (int k = 0; k != KI; k++) {
			printOneTopic(pairs_list_ia, "Independent aspect", k, topN);
			for (int c = 0; c != trainData.C; c++) {
				printOneTopic(pairs_lists_io.get(c), "Collection " + c
						+ " opnion", k, topN);
			}
		}

		double tscore_sa[][][] = this.term_score_sa();
		double tscore_so[][][] = this.term_score_so();
		List<List<List<Entry<String, Double>>>> pairs_lists_sa = new ArrayList<List<List<Entry<String, Double>>>>();
		for (int c = 0; c != trainData.C; c++) {
			pairs_lists_sa.add(sorted_topicwords(tscore_sa[c], KS));
		}
		List<List<List<Entry<String, Double>>>> pairs_lists_so = new ArrayList<List<List<Entry<String, Double>>>>();
		for (int c = 0; c != trainData.C; c++) {
			pairs_lists_so.add(sorted_topicwords(tscore_so[c], KS));
		}

		for (int c = 0; c != trainData.C; c++) {
			for (int k = 0; k != KS; k++) {
				printOneTopic(pairs_lists_sa.get(c), "Collection " + c
						+ " Specific aspect", k, topN);
				printOneTopic(pairs_lists_so.get(c), "Collection " + c
						+ " Specific opnion", k, topN);
			}
		}
	}

	// public void printPlainAllTopics_tscore(int topN) {
	// double tscore_ia[][] = this.term_score_ia();
	// double tscore_io[][][] = this.term_score_io();
	//
	// List<List<Entry<String, Double>>> pairs_list_ia = this
	// .sorted_topicwords(tscore_ia, KI);
	// List<List<List<Entry<String, Double>>>> pairs_lists_io = new
	// ArrayList<List<List<Entry<String, Double>>>>();
	// for (int c = 0; c != trainData.C; c++) {
	// pairs_lists_io.add(sorted_topicwords(tscore_io[c], KI));
	// }
	//
	// for (int k = 0; k != KI; k++) {
	// for (int c = 0; c != trainData.C; c++) {
	// printPlainOneTopic(pairs_list_ia, "\t", k, topN);
	// printPlainOneTopic(pairs_lists_io.get(c), "\n", k, topN);
	// }
	// }
	//
	// double tscore_sa[][][] = this.term_score_sa();
	// double tscore_so[][][] = this.term_score_so();
	// List<List<List<Entry<String, Double>>>> pairs_lists_sa = new
	// ArrayList<List<List<Entry<String, Double>>>>();
	// for (int c = 0; c != trainData.C; c++) {
	// pairs_lists_sa.add(sorted_topicwords(tscore_sa[c], KS));
	// }
	// List<List<List<Entry<String, Double>>>> pairs_lists_so = new
	// ArrayList<List<List<Entry<String, Double>>>>();
	// for (int c = 0; c != trainData.C; c++) {
	// pairs_lists_so.add(sorted_topicwords(tscore_so[c], KS));
	// }
	//
	// for (int c = 0; c != trainData.C; c++) {
	// for (int k = 0; k != KS; k++) {
	// printPlainOneTopic(pairs_lists_sa.get(c), "\t", k, topN);
	// printPlainOneTopic(pairs_lists_so.get(c), "\n", k, topN);
	// }
	// }
	// }

	public void printPlainAllTopics_tscore(int topN) {
		double tscore_ia[][] = this.term_score_ia();
		double tscore_io[][][] = this.term_score_io();

		List<List<Entry<String, Double>>> pairs_list_ia = this
				.sorted_topicwords(tscore_ia, KI);
		List<List<List<Entry<String, Double>>>> pairs_lists_io = new ArrayList<List<List<Entry<String, Double>>>>();
		for (int c = 0; c != trainData.C; c++) {
			pairs_lists_io.add(sorted_topicwords(tscore_io[c], KI));
		}

		double tscore_sa[][][] = this.term_score_sa();
		double tscore_so[][][] = this.term_score_so();
		List<List<List<Entry<String, Double>>>> pairs_lists_sa = new ArrayList<List<List<Entry<String, Double>>>>();
		for (int c = 0; c != trainData.C; c++) {
			pairs_lists_sa.add(sorted_topicwords(tscore_sa[c], KS));
		}
		List<List<List<Entry<String, Double>>>> pairs_lists_so = new ArrayList<List<List<Entry<String, Double>>>>();
		for (int c = 0; c != trainData.C; c++) {
			pairs_lists_so.add(sorted_topicwords(tscore_so[c], KS));
		}

		for (int c = 0; c != trainData.C; c++) {
			for (int k = 0; k != KI; k++) {
				printPlainOneTopic(pairs_list_ia, "\t", k, topN);
				printPlainOneTopic(pairs_lists_io.get(c), "\n", k, topN);
			}

			for (int k = 0; k != KS; k++) {
				printPlainOneTopic(pairs_lists_sa.get(c), "\t", k, topN);
				printPlainOneTopic(pairs_lists_so.get(c), "\n", k, topN);
			}
		}
	}

	public void save_r_z_states(String outdir, String name) {
		BufferedWriter writer = IOUtils.getWriter(outdir + name, "utf-8");
		try {
			for (int m = 0; m != r_assigns.length; m++) {
				for (int s = 0; s != r_assigns[m].length; s++) {
					writer.append(r_assigns[m][s] + ":" + z_assigns[m][s] + " ");
				}
				writer.append("\n");
			}
			writer.flush();
			writer.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public void compute_jsd_io() {
		double[][][] phi_io = this.compute_phi_io();

		for (int k = 0; k != KI; k++) {
			for (int i = 1; i != trainData.C; i++) {
				for (int j = 0; j != i; j++) {
					System.out.print("Independent aspect" + k + " collection "
							+ j + " vs. collection " + i + ":");
					double jsd = MathUtil.jensenShannonDivergence(phi_io[j][k],
							phi_io[i][k]);
					System.out.println(jsd);
				}
			}
		}
	}

	public double updateParmas(int i) {
		double[] zd = new double[KS + KI];
		for (int j = 0; j < z_assigns[i].length; j++) {
			int index = (r_assigns[i][j] == 0) ? z_assigns[i][j]
					: z_assigns[i][j] + KS;

			zd[index] += 1.0 / z_assigns[i].length;
		}

		double mutil = 0;
		for (int k = 0; k < KS + KI; k++) {
			mutil += zd[k] * coeffs[k];
		}

		double x = Math.exp(-lables[i] * mutil);
		double result = (1 + x * Math.exp(lables[i] * zd[i])
				/ z_assigns[i].length)
				/ (1 + x);
		return result;
	}

	/**
	 * 获取逻辑回归系数
	 * 
	 * @return
	 */
	public double[] getCoefficents() {
		List<DataPointPair<Double>> dps = new ArrayList<>();
		for (int i = 0; i < z_assigns.length; i++) {
			double[] sentenceTopics = new double[KS + KI];
			for (int j = 0; j < z_assigns[i].length; j++) {
				int index = (r_assigns[i][j] == 0) ? z_assigns[i][j]
						: z_assigns[i][j] + KS;

				sentenceTopics[index] += 1.0 / z_assigns[i].length;
			}
			DenseVector vec = new DenseVector(sentenceTopics);
			DataPoint dataPoint = new DataPoint(vec, 1.0);
			DataPointPair<Double> dp = new DataPointPair<Double>(dataPoint,
					lables[i]);
			dps.add(dp);
		}
		RegressionDataSet dastaSet = new RegressionDataSet(dps);
		LogisticRegression lr = new LogisticRegression();
		lr.train(dastaSet);
		Vec vec = lr.getCoefficents();
		coeffs = vec.arrayCopy();
		return coeffs;
	}

	public static void main(String args[]) throws IOException {
		for (int i = 0; i != 3; i++) {
			ccMaxEntLDA ccMELDA = new ccMaxEntLDA();
			ccMELDA.loadCorpus("/Users/zuoyuan/Desktop/experiment/train_sample1.txt");
			ccMELDA.initialize();
			ccMELDA.estimate();
			ccMELDA.printPlainAllTopics(20, "cc_output_c1_run" + i + ".txt",
					"cc_output_c2_run" + i + ".txt");
		}
		// ccMELDA.printPlainAllTopics_tscore(20);
		// ccMELDA.printAllTopics_tscore(10);
		// ccMELDA.compute_jsd_io();
		// ccMELDA.save_r_z_states("L:/zuoyuan/MaxEntLDA/event/trampling event/new/",
		// "states_sample2.txt");
	}
}
