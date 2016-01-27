package util;

import java.math.BigInteger;

public class MathUtil {

	public static final double log2 = Math.log(2);

	/**
	 * Returns the KL divergence, K(p1 || p2).
	 * 
	 * The log is w.r.t. base 2.
	 * <p>
	 * 
	 * *Note*: If any value in <tt>p2</tt> is <tt>0.0</tt> then the
	 * KL-divergence is <tt>infinite</tt>. Limin changes it to zero instead of
	 * infinite.
	 * 
	 */
	public static double klDivergence(double[] p1, double[] p2) {

		double klDiv = 0.0;

		for (int i = 0; i < p1.length; ++i) {
			if (p1[i] == 0) {
				continue;
			}
			if (p2[i] == 0.0) {
				continue;
			} // Limin

			klDiv += p1[i] * Math.log(p1[i] / p2[i]);
		}

		return klDiv / log2; // moved this division out of the loop -DM
	}
	
	  /**
     * Returns the Jensen-Shannon divergence.
     */
    public static double jensenShannonDivergence(double[] p1, double[] p2) {
      assert(p1.length == p2.length);
      double[] average = new double[p1.length];
      for (int i = 0; i < p1.length; ++i) {
        average[i] += (p1[i] + p2[i])/2;
      }
      return (klDivergence(p1, average) + klDivergence(p2, average))/2;
    }
    
    public static double jsd(double[] h1, double[] h2) {
        float sum = 0f;
        for (int i = 0; i < h1.length; i++) {
            sum += (h1[i] > 0 ? (h1[i] / 2f) * Math.log((2f * h1[i]) / (h1[i] + h2[i])) : 0) +
                    (h2[i] > 0 ? (h2[i] / 2f) * Math.log((2f * h2[i]) / (h1[i] + h2[i])) : 0);
        }
        return sum;
    }
    
	public static double Cosine(double[] a,double[] b){
		double a_2=0,b_2=0,ab=0;
		for(int i=0;i<a.length;i++){
			a_2 += a[i]*a[i];
			b_2 += b[i]*b[i];
			ab  += a[i]*b[i];
		}
		return ab/(Math.sqrt(a_2)*Math.sqrt(b_2));
	}

	public static int gcd(int a, int b) {
		return BigInteger.valueOf(a).gcd(BigInteger.valueOf(b)).intValue();
	}

	public static int gcd(int[] in) {
		int result = in[0];
		for (int i = 1; i < in.length; i++)
			result = gcd(result, in[i]);
		return result;
	}

	public static int lcm(int a, int b) {
		return a * b / gcd(a, b);
	}

	public static int lcm(int[] in) {
		int result = in[0];
		for (int i = 1; i < in.length; i++)
			result = lcm(result, in[i]);
		return result;
	}

	public static double innerProduct(Double v1[], Double v2[]) {
		if (v1.length != v2.length)
			return 0.0;
		double sum = 0.0;
		double sum1 = 0.0, sum2 = 0.0;
		for (int i = 0; i != v1.length; ++i) {
			sum += v1[i] * v2[i];
			sum1 += v1[i] * v1[i];
			sum2 += v2[i] * v2[i];
		}
		return sum / (Math.sqrt(sum1) * Math.sqrt(sum2));
	}

	public static double innerProduct(double v1[], double v2[]) {
		if (v1.length != v2.length)
			return 0.0;
		double sum = 0.0;
		double sum1 = 0.0, sum2 = 0.0;
		for (int i = 0; i != v1.length; ++i) {
			sum += v1[i] * v2[i];
			sum1 += v1[i] * v1[i];
			sum2 += v2[i] * v2[i];
		}
		return sum / (Math.sqrt(sum1) * Math.sqrt(sum2));
	}

	public static double getMean(double vec[]) {
		double sum = 0.0;
		for (double d : vec) {
			sum += d;
		}
		return sum / vec.length;
	}

	public static Double getMean(Double vec[]) {
		Double sum = 0.0;
		for (Double d : vec) {
			sum += d;
		}
		return sum / vec.length;
	}

	public static double getDeviation(double vec[]) {
		double m = MathUtil.getMean(vec);
		double sum = 0.0;
		for (double d : vec) {
			sum += Math.pow(d - m, 2);
		}
		return Math.sqrt(sum / (vec.length - 1));
	}

	public static Double getDeviation(Double vec[]) {
		Double m = MathUtil.getMean(vec);
		Double sum = 0.0;
		for (Double d : vec) {
			sum += Math.pow(d - m, 2);
		}
		return Math.sqrt(sum / (vec.length - 1));
	}

	public static double getPearsonCorrelation(double v1[], double v2[]) {
		double m1 = MathUtil.getMean(v1);
		double m2 = MathUtil.getMean(v2);
		double d1 = MathUtil.getDeviation(v1);
		double d2 = MathUtil.getDeviation(v2);
		double sum = 0.0;
		for (int i = 0; i != v1.length; i++) {
			sum += ((v1[i] - m1) / d1) * ((v2[i] - m2) / d2);
		}
		return sum / (v1.length - 1);
	}

	public static Double getPearsonCorrelation(Double v1[], Double v2[]) {
		Double m1 = MathUtil.getMean(v1);
		Double m2 = MathUtil.getMean(v2);
		Double d1 = MathUtil.getDeviation(v1);
		Double d2 = MathUtil.getDeviation(v2);
		Double sum = 0.0;
		for (int i = 0; i != v1.length; i++) {
			sum += ((v1[i] - m1) / d1) * ((v2[i] - m2) / d2);
		}
		return sum / (v1.length - 1);
	}

	public static void main(String args[]) {
		double[] d3 = { 0.3, 0.15, 0.175, 0.375 };
		// System.out.println(MathUtil.innerProduct(d1, d2));
		System.out.println(MathUtil.getDeviation(d3));
	}
}
