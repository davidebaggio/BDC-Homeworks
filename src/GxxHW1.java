
import java.util.List;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.*;
import scala.Tuple2;

public class GxxHW1 {

	public static void printInfo(String info) {
		System.out.println("***************************************************************");
		System.out.println(info);
		System.out.println("***************************************************************");

	}

	/*
	 * takes in input an RDD of (point,group) pairs (representing a set U=A∪B),
	 * and a set C
	 * of centroids, and returns the value of the objective function Δ(U,C) =
	 * (1/|U|)∑u∈U(dist(u,C))^2 thus ignoring the demopgraphic groups.
	 */
	public static double MRComputeStandardObjective(JavaPairRDD<Vector, String> pairs, KMeansModel centroids) {
		return pairs.map((pair) -> Vectors.sqdist(pair._1(), centroids.clusterCenters()[centroids.predict(pair._1())]))
				.reduce((x, y) -> x + y) / (double) pairs.count();
	}

	/*
	 * takes in input an RDD of (point,group) pairs (representing points of a set
	 * U=A∪B), and a set C
	 * of centroids, and returns the value of the objective function Φ(A,B,C) =
	 * max{(1/|A|)∑a∈A(dist(a,C))2,(1/|B|)∑b∈B(dist(b,C))2}
	 */
	public static double MRComputeFairObjective(JavaPairRDD<Vector, String> pairs, KMeansModel centroids) {
		return Math.max(pairs.filter((pair) -> pair._2().equals("A")).map((pair) -> Vectors.sqdist(pair._1(),
				centroids.clusterCenters()[centroids.predict(pair._1())])).reduce((x, y) -> x + y)
				/ (double) pairs.filter((pair) -> pair._2().equals("A")).count(),
				pairs.filter((pair) -> pair._2().equals("B")).map((pair) -> Vectors.sqdist(pair._1(),
						centroids.clusterCenters()[centroids.predict(pair._1())])).reduce((x, y) -> x + y)
						/ (double) pairs.filter((pair) -> pair._2().equals("B")).count());
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
			List<Vector> points = pairs.filter((pair) -> (centroids.predict(pair._1()) == j)).map((pair) -> pair._1())
					.collect();
			long countA = pairs.filter((pair) -> (centroids.predict(pair._1()) == j && pair._2().equals("A"))).count();
			long countB = pairs.filter((pair) -> (centroids.predict(pair._1()) == j && pair._2().equals("B"))).count();
			if (points.size() > 0) {
				Vector centroid = centroids.clusterCenters()[i];
				printInfo("Centroid " + i + ": " + centroid + " -> (NA = " + countA + ", NB = " + countB + ")");
			}
		}
	}

	public static void main(String[] args) throws InterruptedException {

		if (args.length != 4) {
			System.err.println("Usage: GxxHW1 <input> <K> <max_iterations> <num_partitions>");
			System.exit(1);
		}
		// org/apache/spark/log4j2-defaults.properties
		Logger.getLogger("org").setLevel(Level.OFF);
		Logger.getLogger("akka").setLevel(Level.OFF);

		SparkConf conf = new SparkConf(true).setAppName("GxxHW1").setMaster("local");
		JavaSparkContext sc = new JavaSparkContext(conf);
		JavaRDD<String> docs = sc.textFile(args[0]).repartition(Integer.parseInt(args[3])).cache();
		JavaRDD<Vector> points = docs.map((line) -> {
			String[] tokens = line.split(",");
			double[] data = new double[tokens.length - 1];
			for (int i = 0; i < tokens.length - 1; i++) {
				data[i] = Double.parseDouble(tokens[i]);
			}
			return Vectors.dense(data);
		});

		KMeansModel centroids = KMeans.train(points.rdd(), Integer.parseInt(args[1]), Integer.parseInt(args[2]));
		JavaPairRDD<Vector, String> pairs = docs.mapToPair((line) -> {
			String[] tokens = line.split(",");
			double[] data = new double[tokens.length - 1];
			for (int i = 0; i < tokens.length - 1; i++) {
				data[i] = Double.parseDouble(tokens[i]);
			}
			return new Tuple2<>(Vectors.dense(data), tokens[tokens.length - 1]);
		});

		double standardObjective = MRComputeStandardObjective(pairs, centroids);
		double fairObjective = MRComputeFairObjective(pairs, centroids);
		printInfo("Value of Δ(U, C) = " + standardObjective);
		printInfo("Value of Φ(A, B, C) = " + fairObjective);
		MRPrintStatistics(pairs, centroids);

		// Thread.sleep(3600 * 1000);
		sc.close();
	}
}
