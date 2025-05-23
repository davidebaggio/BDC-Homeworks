import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Random;

public class G48GEN {
	public static void main(String[] args) {
		if (args.length < 2 || args.length > 3) {
			System.err.println("[USAGE]: G48GEN <N> <K> [outputFile]");
			System.exit(1);
		}

		int N = Integer.parseInt(args[0]);
		int K = Integer.parseInt(args[1]);
		String outputFile = (args.length == 3) ? args[2] : "gen.csv";

		if (N <= 0 || K <= 0) {
			System.err.println("[ERROR]: N and K must be positive integers.");
			System.exit(1);
		}

		int NA = (int) (N * 0.75); // number of A's
		int NB = N - NA; // number of B's

		double spacing = 10.0; // distance between cluster centers on x-axis
		double sigma = 6.0; // Gaussian noise σ
		Random rand = new Random(48);

		try (PrintWriter pw = new PrintWriter(new FileWriter(outputFile))) {

			for (int i = 0; i < NA; i++) {
				double x = sigma * rand.nextGaussian();
				double y = sigma * rand.nextGaussian();
				pw.printf("%.5f,%.5f,A%n", x, y);
				System.out.printf("%.5f,%.5f,A%n", x, y);
			}

			for (int i = 0; i < NB; i++) {
				int c = rand.nextInt(K);
				double cx = c * spacing;
				double cy = 0.0;
				double x = cx + sigma * rand.nextGaussian();
				double y = cy + sigma * rand.nextGaussian();
				pw.printf("%.5f,%.5f,B%n", x, y);
				System.out.printf("%.5f,%.5f,B%n", x, y);
			}

			// System.out.printf("Wrote %d points (%d A, %d B) into %s%n", N, NA, NB,
			// outputFile);

		} catch (IOException e) {
			System.err.println("[ERROR]: writing to file: " + e.getMessage());
			System.exit(2);
		}
	}
}
