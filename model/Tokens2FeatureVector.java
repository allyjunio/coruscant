/*
Author: Jun-Ming Xu (xujm@cs.wisc.edu)

Contact: Jun-Ming Xu (xujm@cs.wisc.edu), Xiaojin Zhu (jerryzhu@cs.wisc.edu)
June 2012

*/

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.ArrayList;
import java.util.HashSet;



public class Tokens2FeatureVector {
	
	HashMap<String, Integer> vocab; 
	
	FeatureVector fv;
	
	public void loadVocab(String file){
		vocab = new HashMap<String, Integer>();
		try{
			BufferedReader br = new BufferedReader(new InputStreamReader(Tokens2FeatureVector.class.getResourceAsStream(file)));
			String line = null;
			int index = 0;
			while ((line = br.readLine()) != null){
				vocab.put(line.trim(), index);
				index++;
			}
		}catch (Exception e){
			e.printStackTrace();
		}
	}
	
	public void covertFeatureVector(ArrayList<String> tokens){
		fv = new FeatureVector(vocab.size());
		for (String t : tokens){
			Integer idx = vocab.get(t);
			if (idx != null)
				fv.increase(idx);
		}
		for (int i = 1 ; i < tokens.size(); i++){
			Integer idx = vocab.get(tokens.get(i-1) + " " + tokens.get(i));
			if (idx != null)
				fv.increase(idx);
		}
		fv.normalize();
	}
	
	public Integer[] getIndexSet(){
		return fv.getIndexSet();
	}
	
	public double[] getValueSet(){
		return fv.getValueSet();
	}
}

class FeatureVector {

	protected HashSet<Integer> index;
	protected double[] values;
	
	public FeatureVector(int size){
		index = new HashSet<Integer>();
		values = new double[size];
	}
	
	public void increase(Integer i){
		index.add(i);
		values[i] += 1.0;
	}
	
	public void normalize(){
		double ss = 0.0;
		for (int i : index)
			ss += values[i] * values[i];
		ss = Math.sqrt(ss);
		for (int i : index)
			values[i] /= ss;
	}
	
	public Integer[] getIndexSet(){
		return index.toArray(new Integer[index.size()]);
	}
	
	public double[] getValueSet(){
		return values;
	}
	
	public String toString() {
		StringBuffer output = new StringBuffer();
		for (int i : index) {
			output.append(i + ":" + values[i] + "\t");
		}
		output.append("\n");
		return output.toString();
	}
}
