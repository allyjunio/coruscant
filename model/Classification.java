/*
Load classification model and use it to classify the input.

Input: each line from stdin is treated as a short document.  It is expected that the input
has been filtered by Enrichment.jar to remove lines without bullying keywords.  This program
will still work if no filtering has been done, but may produce warnings to stderr.

Output: The program runs SVM classification and outputs the predicted label.

Warnings: the program is intended to work on "enriched tweets."  It will try to detect whether an
input line is such a tweet.  If not, it produces a warning message for that line.  it will still 
attempt to classify the line.  However, the classification accuracy in the presence of a warning
can not be guaranteed.

To cite the code:

Learning from bullying traces in social media
Jun-Ming Xu, Kwang-Sung Jun, Xiaojin Zhu, and Amy Bellmore
In North American Chapter of the Association for Computational Linguistics - 
Human Language Technologies (NAACL HLT)
Montreal, Canada, 2012

Author: Jun-Ming Xu (xujm@cs.wisc.edu)

Contact: Jun-Ming Xu (xujm@cs.wisc.edu), Xiaojin Zhu (jerryzhu@cs.wisc.edu)
June 2012

 */

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;

public class Classification {

	static String[] keywords = { "ignored", "pushed", "rumors", "locker",
		"spread", "shoved", "rumor", "teased", "kicked", "crying",
		"bullied", "bully", "bullyed", "bullying", "bullyer", "bulling" };

	static String[] classifier_type = {"trace", "teasing", "author_role",
		"form", "type"}; 
	double[][] w;
	double bias;
	int num_features;
	int num_classes;
	String class_labels[];

	public static void main(String[] args) {
		if (args.length < 1 ||
				!Arrays.asList(classifier_type).contains(args[0].toLowerCase())) {
			System.out.println("You should specify which classifier you want to use "
					+ "(trace, teasing, author_role, form or type). for example:\n"
					+ "java -jar Classification.jar trace ");
			return;
		}

		String line = null;

		ArrayList<String> tokens = null;
		Tokenizer tokenizer = new Tokenizer();

		Tokens2FeatureVector t2v = new Tokens2FeatureVector();
		t2v.loadVocab("model/vocab");

		Classification classifier = new Classification();
		try {
			classifier.loadModel("model/" + args[0].toLowerCase());
		} catch (Exception e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(
					System.in));
			while ((line = br.readLine()) != null) {
				checkInput(line);
				tokens = tokenizer.tokenize(line);
				t2v.covertFeatureVector(tokens);
				//System.out.println(t2v.fv.toString());
				System.out.println(classifier.classify(t2v.getIndexSet(), t2v
						.getValueSet()));
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	// Since the model is trained on the enriched data, the test data should be
	// also in the enriched data 'containing keywords'.
	// Otherwise, the performance is not guaranteed.
	public static boolean checkInput(String text) {
		if (text.length() > 150) {
			System.err.println("(Warning: line longer than 150 characters): " + text);
			return false;
		}
		String lowerCase = text.toLowerCase();
		boolean containKeyword = false;
		for (String k : keywords)
			if (lowerCase.contains(k)) {
				containKeyword = true;
				break;
			}
		if (containKeyword == false) {
			System.err.println("(Warning: line does not contain keywords): " + text);
			return false;
		}
		if (!lowerCase.contains("bull")) {
			System.err.println("(Warning: line does not contain string \"bull\"): " + text);
			return false;
		}
		if (text.contains("RT")) {
			System.err.println("(Warning: line contains \"RT\", retweet?): " + text);
			return false;
		}
		return true;
	}

	public boolean loadModel(String file) throws Exception {
		BufferedReader br = new BufferedReader(new InputStreamReader(
				Classification.class.getResourceAsStream(file)));
		String line = br.readLine();  // ignore first line

		line = br.readLine(); // nr_class
		String[] fields = line.split(" ");
		if (!fields[0].equals("nr_class")) {
			return false;
		}
		num_classes = Integer.parseInt(fields[1]);

		line = br.readLine(); // label
		fields = line.split(" ");
		if (!fields[0].equals("label") || fields.length < num_classes + 1) {
			return false;
		}
		class_labels = new String[num_classes];
		for (int i = 1; i <= num_classes; i++) {
			class_labels[i-1] = fields[i];
		}

		line = br.readLine(); // nr_feature
		fields = line.split(" ");
		if (!fields[0].equals("nr_feature")) {
			return false;
		}
		num_features = Integer.parseInt(fields[1]);

		line = br.readLine(); // bias
		fields = line.split(" ");
		if (!fields[0].equals("bias")) {
			return false;
		}
		bias = Double.parseDouble(fields[1]);

		line = br.readLine();
		if (!line.equals("w")) {
			return false;
		}

		// parse weights
		int num_weights = num_classes;
		if (num_classes == 2) {
			num_weights = 1;
		} 
		w = new double[num_weights][num_features];
		for (int f = 0; f < num_features; f++) {
			line = br.readLine();
			if (line == null) {
				return false;
			}
			fields = line.split(" ");
			if (fields.length != num_weights)
				return false;
			for (int i = 0; i < num_weights; i++) {
				w[i][f] = Double.parseDouble(fields[i]);
			}
		}
		return true;
	}

	public String classify(Integer[] index, double[] value) {
		if (num_classes == 2) {
			double margin = 0.0;
			for (int i : index) {
				margin += w[0][i] * value[i];
			}
			return margin > 0 ? class_labels[0] : class_labels[1];
		}
		// multiple classes
		double[] margins = new double[num_classes];
		for (int i : index) {
			for (int c = 0; c < num_classes; c++) {
				margins[c] += w[c][i] * value[i]; 
			}
		}
		double max_margin = Double.NEGATIVE_INFINITY;
		int idx = -1;
		for (int c = 0; c < num_classes; c++) {
			if (margins[c] > max_margin) {
				max_margin = margins[c];
				idx = c;
			}
		}
		return class_labels[idx];
	}
}
