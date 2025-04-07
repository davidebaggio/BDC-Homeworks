
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

public class GxxHW1 {

	public static void printInfo(String info) {
		// System.out.println("***************************************************************");
		System.out.println(info);
		// System.out.println("***************************************************************");

	}

	public static Tuple2<Integer, Double> closestCentroidDist(Vector point, KMeansModel centroids) {
		int centroid = 0;
		double dist = Double.MAX_VALUE;
		for (int i = 0; i < centroids.k(); i++) {
			double newDist = Vectors.sqdist(point, centroids.clusterCenters()[i]);
			if (newDist < dist) {
				dist = newDist;
				centroid = i;
			}
		}
		return new Tuple2<Integer, Double>(centroid, dist);
	}

	/*
	 * takes in input an RDD of (point,group) pairs (representing a set U=A∪B),
	 * and a set C
	 * of centroids, and returns the value of the objective function Δ(U,C) =
	 * (1/|U|)∑u∈U(dist(u,C))^2 thus ignoring the demopgraphic groups.
	 */
	public static double MRComputeStandardObjective(JavaPairRDD<Vector, String> pairs, KMeansModel centroids) {
		return pairs.map(
				(pair) -> closestCentroidDist(pair._1(), centroids)._2().doubleValue())
				.reduce((x, y) -> x + y) / (double) pairs.count();
	}

	/*
	 * takes in input an RDD of (point,group) pairs (representing points of a set
	 * U=A∪B), and a set C
	 * of centroids, and returns the value of the objective function Φ(A,B,C) =
	 * max{(1/|A|)∑a∈A(dist(a,C))2,(1/|B|)∑b∈B(dist(b,C))2}
	 */
	public static double MRComputeFairObjective(JavaPairRDD<Vector, String> pairs, KMeansModel centroids) {
		double aCount = (double) pairs.filter((pair) -> pair._2().equals("A")).count();
		double bCount = pairs.count() - aCount;
		if (aCount == 0 || bCount == 0) {
			return 0;
		}
		return Math.max(
				pairs.filter((pair) -> pair._2().equals("A"))
						.map((pair) -> closestCentroidDist(pair._1(), centroids)._2().doubleValue())
						.reduce((x, y) -> x + y) / aCount,
				pairs.filter((pair) -> pair._2().equals("B"))
						.map((pair) -> closestCentroidDist(pair._1(), centroids)._2().doubleValue())
						.reduce((x, y) -> x + y) / bCount);
	}

	/*
	 * takes in input an RDD of (point,group) pairs (representing points of a set
	 * U=A∪B), and a set C
	 * of centroids, and computes and prints the triplets (ci,NAi,NBi), for
	 * 1≤i≤K=|C| , where ci is the i -th centroid in C , and NAi,NBi are the numbers
	 * of points of A and B , respectively, in the cluster Ui centered in ci
	 */
	public static void MRPrintStatistics(JavaPairRDD<Vector, String> pairs, KMeansModel centroids) {
		for (int i = 0; i < centroids.k(); i++) {
			final Integer j = new Integer(i);
			long countA = pairs
					.filter((
							pair) -> (closestCentroidDist(pair._1(), centroids)._1().compareTo(j) == 0
									&& pair._2().equals("A")))
					.count();
			long countB = pairs
					.filter((
							pair) -> (closestCentroidDist(pair._1(), centroids)._1().compareTo(j) == 0
									&& pair._2().equals("B")))
					.count();
			Vector centroid = centroids.clusterCenters()[i];
			printInfo(
					"i = " + i + ", center = " + centroid + ", NA" + i + " = " + countA + ", NB" + i + " = " + countB);
		}
	}

	public static void main(String[] args) throws InterruptedException {

		if (args.length != 4) {
			System.err.println("Usage: GxxHW1 <input> <num_partitions> <num_clusters> <num_iterations>");
			System.exit(1);
		}
		String inputPath = args[0];
		final int L = Integer.parseInt(args[1]);
		final int K = Integer.parseInt(args[2]);
		final int M = Integer.parseInt(args[3]);
		if (K <= 0 || M <= 0 || L <= 0) {
			System.err.println("K, M and L must be positive integers");
			System.exit(1);
		}

		// org/apache/spark/log4j2-defaults.properties
		Logger.getLogger("com").setLevel(Level.OFF);
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);
		Logger.getRootLogger().setLevel(Level.OFF);

		SparkConf conf = new SparkConf(true).setAppName("GxxHW1").setMaster("local");
		JavaSparkContext sc = new JavaSparkContext(conf);
		sc.setLogLevel("OFF");
		JavaRDD<String> docs = sc.textFile(inputPath);

		JavaPairRDD<Vector, String> inputPoints = docs.mapToPair((line) -> {
			String[] tokens = line.split(",");
			double[] data = new double[tokens.length - 1];
			for (int i = 0; i < tokens.length - 1; i++) {
				data[i] = Double.parseDouble(tokens[i]);
			}
			return new Tuple2<>(Vectors.dense(data), tokens[tokens.length - 1]);
		}).repartition(L).cache();

		KMeansModel centroids = KMeans.train(inputPoints.map((pair) -> pair._1()).rdd(), K, M, "k-means||", 48);

		double standardObjective = MRComputeStandardObjective(inputPoints, centroids);
		double fairObjective = MRComputeFairObjective(inputPoints, centroids);

		long N = inputPoints.count();
		long NA = inputPoints.filter((pair) -> pair._2().equals("A")).map((pair) -> pair._1()).count();
		long NB = N - NA;

		printInfo("Input file = " + inputPath + ", L = " + L + ", K = " + K + ", M = " + M);
		printInfo("N = " + N + ", NA = " + NA + ", NB = " + NB);
		printInfo("Delta(U, C) = " + standardObjective);
		printInfo("Phi(A, B, C) = " + fairObjective);
		MRPrintStatistics(inputPoints, centroids);

		//Thread.sleep(3600 * 1000);
		sc.close();
	}
}
