all:
	javac Enrichment.java
	jar cfm Enrichment.jar manifest_e.txt Enrichment.class
	javac Classification.java Tokenizer.java Tokens2FeatureVector.java
	jar cfm Classification.jar manifest_c.txt Classification.class Tokenizer.class Tokens2FeatureVector.class FeatureVector.class model/vocab model/author_role model/form model/teasing model/trace model/type

clean:
	rm *.class
