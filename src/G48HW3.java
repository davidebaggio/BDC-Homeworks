import org.apache.spark.SparkConf;
import org.apache.spark.api.java.StorageLevels;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaStreamingContext;

import java.util.*;
import java.util.concurrent.Semaphore;

@SuppressWarnings("deprecation")
public class G48HW3 {

	private static final int P = 8191;

	// Hash function class
	static class HashFunction {
		private final int a, b, C;
		private final Random rand = new Random();

		public HashFunction(int C) {
			this.C = C;
			this.a = rand.nextInt(P - 1) + 1;
			this.b = rand.nextInt(P);
		}

		public int hash(long x) {
			return (int) (((a * x + b) % P) % C);
		}
	}

	public static void main(String[] args) throws Exception {
		if (args.length != 5) {
			throw new IllegalArgumentException("[USAGE]: <Port>, <T> <D> <W> <K>");
		}

		int portExp = Integer.parseInt(args[0]);
		int T = Integer.parseInt(args[1]);
		int D = Integer.parseInt(args[2]);
		int W = Integer.parseInt(args[3]);
		int K = Integer.parseInt(args[4]);

		SparkConf conf = new SparkConf(true).setMaster("local[*]").setAppName("G48HW3");

		JavaStreamingContext sc = new JavaStreamingContext(conf, Durations.milliseconds(100));
		sc.sparkContext().setLogLevel("OFF");

		Semaphore stoppingSemaphore = new Semaphore(1);
		stoppingSemaphore.acquire();

		System.out.println("Receiving data from port = " + portExp);
		System.out.println("Threshold = " + T);

		long[] streamLength = new long[] { 0 };
		HashMap<Long, Long> histogram = new HashMap<>();
		int[][] CM = new int[D][W];
		int[][] CS = new int[D][W];
		HashFunction[] hCM = new HashFunction[D];
		HashFunction[] hCS = new HashFunction[D];
		HashFunction[] gCS = new HashFunction[D];

		for (int i = 0; i < D; i++) {
			hCM[i] = new HashFunction(W);
			hCS[i] = new HashFunction(W);
			gCS[i] = new HashFunction(2);
		}

		sc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevels.MEMORY_AND_DISK)
				.foreachRDD((batch, time) -> {
					if (streamLength[0] < T) {

						List<String> items = batch.collect();
						streamLength[0] += items.size();

						for (String s : items) {
							long x;
							try {
								x = Long.parseLong(s);
							} catch (NumberFormatException e) {
								continue;
							}

							histogram.put(x, histogram.getOrDefault(x, 0L) + 1);

							for (int i = 0; i < D; i++) {
								int idxCM = hCM[i].hash(x);
								CM[i][idxCM] += 1;

								int idxCS = hCS[i].hash(x);
								int sign = gCS[i].hash(x) == 0 ? -1 : 1;
								CS[i][idxCS] += sign;
							}
						}

						if (streamLength[0] >= T) {
							stoppingSemaphore.release();
						}
					}
				});

		// MANAGING STREAMING SPARK CONTEXT
		System.out.println("Starting streaming engine");
		sc.start();
		System.out.println("Waiting for shutdown condition");
		stoppingSemaphore.acquire();
		System.out.println("Stopping the streaming engine");
		sc.stop(false, false);
		System.out.println("Streaming engine stopped");
		sc.close();

		List<Map.Entry<Long, Long>> sorted = new ArrayList<>(
				histogram.entrySet());
		sorted.sort((a, b) -> Long.compare(b.getValue(), a.getValue()));
		long phiK = sorted.get(Math.min(K - 1, sorted.size() - 1)).getValue();

		List<Long> topK = new ArrayList<>();
		for (Map.Entry<Long, Long> e : sorted) {
			if (e.getValue() >= phiK) {
				topK.add(e.getKey());
			} else {
				break;
			}
		}

		double totalErrorCM = 0.0;
		double totalErrorCS = 0.0;

		Map<Long, Long> cmEstimates = new HashMap<>();
		Map<Long, Long> csEstimates = new HashMap<>();

		for (Long x : topK) {
			long estCM = Long.MAX_VALUE;
			for (int i = 0; i < D; i++) {
				estCM = Math.min(estCM, CM[i][hCM[i].hash(x)]);
			}
			cmEstimates.put(x, estCM);
			totalErrorCM += Math.abs(histogram.get(x) - estCM) / (double) histogram.get(x);

			List<Long> estimates = new ArrayList<>();
			for (int i = 0; i < D; i++) {
				int sign = gCS[i].hash(x) == 0 ? -1 : 1;
				estimates.add((long) CS[i][hCS[i].hash(x)] * sign);
			}
			Collections.sort(estimates);
			long estCS = estimates.get(D / 2);
			csEstimates.put(x, estCS);
			totalErrorCS += Math.abs(histogram.get(x) - estCS) / (double) histogram.get(x);
		}

		double avgErrorCM = totalErrorCM / topK.size();
		double avgErrorCS = totalErrorCS / topK.size();

		// COMPUTE AND PRINT FINAL STATISTICS
		System.out.printf("Port = %d T = %d D = %d W = %d K = %d%n", portExp, T, D, W, K);
		System.out.println("Number of processed items = " + streamLength[0]);
		System.out.println("Number of distinct items  = " + histogram.size());
		System.out.println("Number of Top-K Heavy Hitters = " + topK.size());
		System.out.printf("Avg Relative Error for Top-K Heavy Hitters with CM = %f%n", avgErrorCM);
		System.out.printf("Avg Relative Error for Top-K Heavy Hitters with CS = %f%n", avgErrorCS);

		if (K <= 10) {
			System.out.println("Top-K Heavy Hitters:");
			topK.sort(Long::compareTo); // Sort by item value
			for (Long x : topK) {
				System.out.printf("Item %d True Frequency = %d Estimated Frequency with CM = %d%n",
						x, histogram.get(x), cmEstimates.get(x));
			}
		}
	}
}