/*
This program filters stdin to find lines containing (roughly) bull* keywords.

The exact rules are a bit strange due to historical reasons:

1. The line must contain the following keywords (case insensitive):
"ignored", "pushed", "rumors", "locker", "spread", "shoved", "rumor", "teased", "kicked", 
"crying", "bullied", "bully", "bullyed", "bullying", "bullyer", "bulling"
Note that "bullies" was NOT included due to our omission.

2. Among those, the line must then contain the string "bull".  This was intended to 
find bully, bullying, bullied and other variants.  However, a line that contains, 
say, "redbull" AND "kicked" would survive our filter too.  Fortunately, this does not happen 
very often.

3. The line cannot contain "RT" which in Twitter means retweet.

Lines surviving the above filtering process are printed to stdout.

Although the rules are not perfect, they exactly replicate the process in the NAACL'12 paper.

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

public class Enrichment {

	static String[] keywords = { "ignored", "pushed", "rumors", "locker",
			"spread", "shoved", "rumor", "teased", "kicked", "crying",
			"bullied", "bully", "bullyed", "bullying", "bullyer", "bulling"};

	public static void main(String[] args) {
		String line = null;
		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
			while ((line = br.readLine()) != null) {
				boolean containKeyword = false;
				String lowerCase = line.toLowerCase();
				for (String k : keywords)
					if (lowerCase.contains(k)) {
						containKeyword = true;
						break;
					}
				if (containKeyword == true && lowerCase.contains("bull")
						&& !line.contains("RT")) {
					System.out.println(line);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
