Bullying Trace Classifier V2.0
Feb 2016

This package includes two modules for recognizing bullying traces, and classifying bullying traces with different tasks.  See comments in source files for more details.

--------------------------------------------------------------
Step 1: 
Create a plain text file with the text you want to classify.  Each line is treated as a mini document.
See "test.txt" for an example.

--------------------------------------------------------------
Step 2: 
Run Enrichment.jar to filter the file so as to find lines containing (roughly) bull* keywords. This produces what's called "enriched data".

Example usage:
   cat test.txt | java -jar Enrichment.jar > test_filtered.txt

The file "test.txt" contains 7 example tweets.  The code filters out one tweet "Lauren is a fat cow MOO BITCH" that doesn't contain any of the keywords.

--------------------------------------------------------------
Step 3: 
Run Classification.jar which takes enriched data and classifies each line. You need to specify which task you woule like to predict ("trace", "teasing", "author_role", "form", "type").

Example usage:
To predict if a tweet is a bullying trace or not:
   cat test_filtered.txt | java -jar Classification.jar trace > bullying_trace.txt
   
To predict the author role of the bullying traces: 
   cat bullying_trace.txt | java -jar Classification.jar author_role > author_role.txt


--------------------------------------------------------------
Citation:

Learning from bullying traces in social media
Jun-Ming Xu, Kwang-Sung Jun, Xiaojin Zhu, and Amy Bellmore
In North American Chapter of the Association for Computational Linguistics - Human Language Technologies (NAACL HLT)
Montreal, Canada, 2012

Contact: Jun-Ming Xu (xujm@cs.wisc.edu), Xiaojin Zhu (jerryzhu@cs.wisc.edu)
