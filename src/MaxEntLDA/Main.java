package MaxEntLDA;

import java.util.Arrays;

import jsat.classifiers.Classifier;
import jsat.classifiers.DataPoint;
import jsat.classifiers.DataPointPair;
import jsat.linear.Vec;
import jsat.regression.LogisticRegression;
import jsat.regression.RegressionDataSet;
import jsat.linear.DenseVector;

public class Main {
	public static void main(String[] args) {
		DenseVector vec = new DenseVector(0);
		DataPoint dataPoint = new DataPoint(vec, 1.0);
		DataPointPair<Double> dp = new DataPointPair<Double>(dataPoint, 1.0);
		RegressionDataSet dastaSet = new RegressionDataSet(Arrays.asList(dp));
		LogisticRegression lr = new LogisticRegression();
		lr.train(dastaSet);

	}
}
