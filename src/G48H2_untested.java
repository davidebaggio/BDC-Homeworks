import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;

public class G48HW2 {

    public static void printInfo(String info) {
        System.out.println(info);
    }

    public static Tuple2<Integer, Double> closestCentroidDist(Vector point, KMeansModel centroids) {
        double minDist = Double.MAX_VALUE;
        int closestCentroid = 0;
        for (int i = 0; i < centroids.k(); i++) {
            double dist = Vectors.sqdist(point, centroids.clusterCenters()[i]);
            if (dist < minDist) {
                minDist = dist;
                closestCentroid = i;
            }
        }
        return new Tuple2<>(closestCentroid, minDist);
    }

    public static double MRComputeFairObjective(JavaPairRDD<Vector, String> pairs, KMeansModel centroids) {
        double aCount = (double) pairs.filter((pair) -> pair._2().equals("A")).count();
        double bCount = pairs.count() - aCount;
        if (aCount == 0 || bCount == 0) {
            return 0;
        }
        double aAvgDist = pairs.filter((pair) -> pair._2().equals("A"))
                .map((pair) -> closestCentroidDist(pair._1(), centroids)._2().doubleValue())
                .reduce((x, y) -> x + y) / aCount;
        double bAvgDist = pairs.filter((pair) -> pair._2().equals("B"))
                .map((pair) -> closestCentroidDist(pair._1(), centroids)._2().doubleValue())
                .reduce((x, y) -> x + y) / bCount;
        return Math.max(aAvgDist, bAvgDist);
    }

    public static Vector add(Vector a, Vector b) {
        return Vectors.dense(a.toArray()).plus(Vectors.dense(b.toArray()));
    }

    public static Vector divide(Vector a, double b) {
        return Vectors.dense(a.toArray()).scale(1.0 / b);
    }

    public static Vector multiply(Vector a, double b) {
        return Vectors.dense(a.toArray()).scale(b);
    }

    public static Vector zeroVector(int dim) {
        return Vectors.dense(new double[dim]);
    }

    public static KMeansModel centroidSelection(JavaPairRDD<Integer, Tuple2<Vector, String>> clusteredPoints,
                                              KMeansModel clusters, int k) {

        int dim = clusters.clusterCenters()[0].size();
        long sizeA = clusteredPoints.filter((pair) -> pair._2()._2().equals("A")).count();
        long sizeB = clusteredPoints.filter((pair) -> pair._2()._2().equals("B")).count();

        JavaPairRDD<Tuple2<Integer, String>, Tuple2<Vector, Long>> paired = clusteredPoints.mapToPair(pair -> {
            Tuple2<Integer, String> key = new Tuple2<>(pair._1(), pair._2()._2());
            return new Tuple2<>(key, new Tuple2<>(pair._2()._1(), 1L));
        }).cache();

        JavaPairRDD<Tuple2<Integer, String>, Tuple2<Vector, Long>> aggregated = paired
                .reduceByKey((a, b) -> new Tuple2<>(add(a._1(), b._1()), a._2() + b._2())).cache();

        JavaPairRDD<Integer, Tuple2<Vector, Long>> aggregatedA = aggregated
                .filter(pair -> pair._1()._2().equals("A"))
                .mapToPair(pair -> new Tuple2<>(pair._1()._1(), pair._2())).cache();

        JavaPairRDD<Integer, Tuple2<Vector, Long>> aggregatedB = aggregated
                .filter(pair -> pair._1()._2().equals("B"))
                .mapToPair(pair -> new Tuple2<>(pair._1()._1(), pair._2())).cache();

        JavaPairRDD<Integer, Tuple2<Tuple2<Vector, Long>, Tuple2<Vector, Long>>> joinedStats = aggregatedA
                .fullOuterJoin(aggregatedB)
                .mapValues(v -> new Tuple2<>(
                        v._1().orElse(new Tuple2<>(zeroVector(dim), 0L)),
                        v._2().orElse(new Tuple2<>(zeroVector(dim), 0L))))
                .cache();

        JavaPairRDD<Integer, Tuple2<Vector, Vector>> mus = joinedStats.mapValues(stats -> {
            Vector muA = stats._1()._2() > 0 ? divide(stats._1()._1(), (double) stats._1()._2()) : zeroVector(dim);
            Vector muB = stats._2()._2() > 0 ? divide(stats._2()._1(), (double) stats._2()._2()) : zeroVector(dim);
            return new Tuple2<>(muA, muB);
        }).cache();

        Map<Integer, Tuple2<Vector, Vector>> muMap = mus.collectAsMap();
        double[] alpha = new double[k];
        double[] beta = new double[k];
        Vector[] muaArr = new Vector[k];
        Vector[] mubArr = new Vector[k];
        double[] l = new double[k];

        for (int j = 0; j < k; j++) {
            final int i = j;
            long sizeAi = joinedStats.lookup(i).get(0)._1()._2();
            long sizeBi = joinedStats.lookup(i).get(0)._2()._2();

            alpha[i] = (double) sizeAi / sizeA;
            beta[i] = (double) sizeBi / sizeB;

            Tuple2<Vector, Vector> mu = muMap.get(i);
            muaArr[i] = mu._1();
            mubArr[i] = mu._2();

            l[i] = Math.sqrt(Vectors.sqdist(muaArr[i], mubArr[i]));
        }

        double deltaA = clusteredPoints.filter((pair) -> (pair)._2()._2().equals("A"))
                .map((pair) -> {
                    int idx = pair._1();
                    return Vectors.sqdist(pair._2()._1(), muaArr[idx]);
                }).reduce((a, b) -> a + b);

        double deltaB = clusteredPoints.filter((pair) -> (pair)._2()._2().equals("B"))
                .map((pair) -> {
                    int idx = pair._1();
                    return Vectors.sqdist(pair._2()._1(), mubArr[idx]);
                }).reduce((a, b) -> a + b);

        double fixedA = deltaA / sizeA;
        double fixedB = deltaB / sizeB;

        double[] x = computeVectorX(fixedA, fixedB, alpha, beta, l, k);
        Vector[] centroids = new Vector[k];
        for (int i = 0; i < k; i++) {
            centroids[i] = divide(add(multiply(muaArr[i], l[i] - x[i]), multiply(mubArr[i], x[i])), l[i]);
        }
        return new KMeansModel(centroids);
    }

    public static double[] computeVectorX(double fixedA, double fixedB, double[] alpha, double[] beta, double[] ell,
                                          int K) {
        double gamma = 0.5;
        double[] xDist = new double[K];
        double fA, fB;
        double power = 0.5;
        int T = 10;
        for (int t = 1; t <= T; t++) {
            fA = fixedA;
            fB = fixedB;
            power = power / 2;
            for (int i = 0; i < K; i++) {
                double temp = (1 - gamma) * beta[i] * ell[i] / (gamma * alpha[i] + (1 - gamma) * beta[i]);
                xDist[i] = temp;
                fA += alpha[i] * temp * temp;
                temp = (ell[i] - temp);
                fB += beta[i] * temp * temp;
            }
            if (fA == fB) {
                break;
            }
            gamma = (fA > fB) ? gamma + power : gamma - power;
        }
        return xDist;
    }

    public static Vector[] MRFairLloyd(JavaPairRDD<Vector, String> pairs, int K, int M) {
        KMeansModel centroids = KMeans.train(pairs.map((pair) -> pair._1()).rdd(), K, 0, "k-means||", 48);

        for (int m = 0; m < M; m++) {
            final Vector[] centroidsVector = centroids.clusterCenters();
            JavaPairRDD<Integer, Tuple2<Vector, String>> clusteredPoints = pairs.mapToPair((pair) -> {
                int closest = closestCentroidDist(pair._1(), new KMeansModel(centroidsVector))._1();
                return new Tuple2<>(closest, pair);
            }).cache();
            centroids = centroidSelection(clusteredPoints, new KMeansModel(centroidsVector), K);
            clusteredPoints.unpersist();
        }
        return centroids.clusterCenters();
    }

    public static void main(String[] args) {
        if (args.length != 4) {
            System.err.println("[USAGE]: G48HW2 <input_file> <L> <K> <M>");
            System.exit(1);
        }
        String inputPath = args[0];
        final int L = Integer.parseInt(args[1]);
        final int K = Integer.parseInt(args[2]);
        final int M = Integer.parseInt(args[3]);
        if (K <= 0 || M <= 0 || L <= 0) {
            System.err.println("[ERROR]: L, K and M must be positive integers");
            System.exit(1);
        }

        Logger.getLogger("com").setLevel(Level.OFF);
        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        Logger.getRootLogger().setLevel(Level.OFF);

        SparkConf conf = new SparkConf(true).setAppName("G48HW2");
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

        long N = inputPoints.count();
        long NA = inputPoints.filter((pair) -> pair._2().equals("A")).map((pair) -> pair._1()).count();
        long NB = N - NA;

        long startTime = System.currentTimeMillis();
        KMeansModel centroidsStandard = KMeans.train(inputPoints.map((pair) -> pair._1()).rdd(), K, M, "k-means||", 48);
        long standardTime = System.currentTimeMillis();
        KMeansModel centroidLloyd = new KMeansModel(MRFairLloyd(inputPoints, K, M));
        long lloydTime = System.currentTimeMillis();
        double standardObjective = MRComputeFairObjective(inputPoints, centroidsStandard);
        long stdObjectiveTime = System.currentTimeMillis();
        double fairLloydObjective = MRComputeFairObjective(inputPoints, centroidLloyd);
        long lloydObjectiveTime = System.currentTimeMillis();

        printInfo("Input file = " + inputPath + ", L = " + L + ", K = " + K + ", M = " + M);
        printInfo("N = " + N + ", NA = " + NA + ", NB = " + NB);
        printInfo("Fair Objective with Standard Centers = " + standardObjective);
        printInfo("Fair Objective with Fair Centers = " + fairLloydObjective);
        printInfo("Time to compute standard centers = " + (standardTime - startTime) + "ms");
        printInfo("Time to compute fair centers = " + (lloydTime - standardTime) + "ms");
        printInfo("Time to compute objective with standard centers = " + (stdObjectiveTime - lloydTime) + "ms");
        printInfo("Time to compute objective with fair centers = " + (lloydObjectiveTime - stdObjectiveTime) + "ms");

        sc.close();
    }
}
