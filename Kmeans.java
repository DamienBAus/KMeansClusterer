package ml.classifier;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import java.util.Stack;
import java.util.TreeSet;

import nlp.nicta.filters.StopWordChecker;
import text.WordCount;
import util.DocUtils;
import util.FileFinder;
/**
 * This class implements the K-means algorithm. The variable parameters are given in the main
 * function. The stopping criteria condition is that it has performed 100 iterations, or that
 * the total number of documents assigned to each cluster remains constant for a set number of
 * iterations.
 * @author Damien Beard
 *
 */
public class Kmeans {
	
	static HashMap<Integer, Document> documents; //List of ID's -> Documents
	static HashMap<Integer, Centroid> centroids; //List of ID's -> Documents
	static HashMap<String,Integer> dictionary;  //List of words -> index value
	static boolean debug = true;
	
	public static void main(String[] args) {
		
		//Alter these variables for examination
		int K = 3;
		String filePath = "data/blog_data_test/";
		
		performKMeansAlgorithm(K, filePath);
	}
	
	static void performKMeansAlgorithm(int K, String inputFilePath){
		// Instantiate document & dictionary lists
		documents = new HashMap<Integer, Document>();
		dictionary = new HashMap<String,Integer>();
		centroids = new HashMap<Integer, Centroid>();
		
		// Step 1. Read in documents, populate document & dictionary maps
		int documentCount = 0;
		
		// Get list of data files
		ArrayList<File> files = FileFinder.GetAllFiles(inputFilePath, "", true);
		if (debug) System.out.println("Found " + files.size() + " files.");
		Extractor extract = new Extractor(1000,true,debug);
		for (File f : files) {
			HashMap<String,Integer> wordsAndCounts = extract.getWordsAndCounts(f);
			Document doc = new Document(documentCount, f.getName());
			// For each (non-stop) word in the document
			for(String word : wordsAndCounts.keySet()) {
				
				// Either get each word from the dictionary, or put add it to it
				int wordIndex;
				if (dictionary.containsKey(word)) wordIndex = dictionary.get(word);
				else {
					wordIndex = dictionary.size();
					dictionary.put(word, wordIndex);
				}
				
				// Add the word index and its frequency to the document vector
				doc.addword(wordIndex, wordsAndCounts.get(word));
				
			}
			doc.getWordSet().normalise();
			documents.put(documentCount, doc);
			documentCount++;
		}
		if (debug) System.out.println("Processed "+documents.size()+" documents yo.");
		

		// Step 2. Create and assign initial centroids
		
		//Remember all random values used, to make sure we don't use the same value twice
		HashMap<Integer, Boolean> usedVals = new HashMap<Integer, Boolean>();
		HashMap<Integer, Integer> clusterTotals = new HashMap<Integer, Integer>();
		Random r = new Random();
		int randomDoc;
		
		//Assign centroid to random documents
		for (int i=0;i<K;i++){
			randomDoc = r.nextInt(documentCount);
			while (usedVals.containsKey(randomDoc)) randomDoc = r.nextInt(documentCount);
			centroids.put(i, new Centroid(i, documents.get(randomDoc).getWordSet(), documents,dictionary));
			if (debug) System.out.println("Centroid "+i+" assigned document "+randomDoc);
			usedVals.put(randomDoc, true);
			clusterTotals.put(i, 0);
		}
		
		int iterations = 0;
		boolean converged = false;
		int convergeCount = 0; //When totals don't change for convergeLimit iterations, it is considered converged
		int convergeLimit = 15;
		
		// Step 3. Find each document's closest centroid
		while (iterations <100 && !converged) {
			
			for(Integer centID : centroids.keySet()) centroids.get(centID).resetDocs();
			
			for(Integer docID : documents.keySet()){
				double highestSimilarity = 0.0, currentSimilarity = 0.0;
				Integer closestCentroid = 0;
				for(Integer centID : centroids.keySet()) {
					currentSimilarity = cosineSimilarity(centroids.get(centID), documents.get(docID));
					if (currentSimilarity>highestSimilarity) {
						highestSimilarity = currentSimilarity;
						closestCentroid = centID;
					}
				}
				documents.get(docID).setDistanceToCentroid(highestSimilarity);
				centroids.get(closestCentroid).addDocs(docID);
			}
			
			if (debug) {
				for(Integer centID : centroids.keySet()) 
					System.out.println("Num docs attached to centroid "+centID+" are "+centroids.get(centID).docsAssigned.size());
			}
			
			// Check to see if the totals have changed since last time
			boolean changedClusterTotal = false;
			for(Integer centID : centroids.keySet()) {
				if (centroids.get(centID).docsAssigned.size()!=clusterTotals.get(centID)) changedClusterTotal = true;
				clusterTotals.put(centID, centroids.get(centID).docsAssigned.size());
			}
			if (!changedClusterTotal) convergeCount++;
			else convergeCount = 0;
			
			if (convergeCount>convergeLimit-1) converged = true;
			
			// Step 4. Recalculate the vector values of the centroid
			for(Integer centID : centroids.keySet()) {
				centroids.get(centID).recalculateCentroid();
			}
			iterations++;
		}
		
		if (debug) System.out.println("Iterations = "+iterations);
		// Step 5. Return top 5 documents for each centroid
		
		for(Integer centID : centroids.keySet()) {
			centroids.get(centID).recalculateCentroid();
			Stack<Integer> top5 = centroids.get(centID).getTopFive();
			System.out.println("\nCluster "+centID);
			while (top5.size()>0) {
				System.out.println(documents.get(top5.pop()).filename);
			}
		}
		
	}

	// Work out the cosine similarity between two documents
	static double cosineSimilarity(Centroid cent, Document doc) {
		// Only iterate over the smaller of the two word vectors
		HashMap<Integer,Double> smallerWordSet, largerWordSet;
		if (cent.getAllWords().size()<doc.getWordSet().getAllWords().size()) {
			smallerWordSet = cent.getAllWords();
			largerWordSet = doc.getWordSet().getAllWords();
		} else {
			largerWordSet = cent.getAllWords();
			smallerWordSet = doc.getWordSet().getAllWords();
		}
		
		double numerator = 0.0;
		for(Integer word : smallerWordSet.keySet()) {
			if (largerWordSet.containsKey(word)) {
				numerator += smallerWordSet.get(word)*largerWordSet.get(word);
			}
		}
		
		double denominator = Math.sqrt(cent.getNormalisationFactor())*
						     	Math.sqrt(doc.getWordSet().getNormalisationFactor());
		
		return (numerator/denominator); //Cosine similarity of centroid and document
	}

}

/**
 * Each file is stored as a document. A document is identifiable by its ID. It contains
 * a list of words, which are retrievable with getWordSet().
 * @author Damien Beard
 *
 */
class Document {
	WordSet words;
	int ID;
	String filename;
	double distanceToCentroid;
	
	Document(int ID, String name){
		this.ID = ID;
		filename = name;
		words = new WordSet();
	}
	
	void addword(Integer word, Integer freq){
		words.put(word,freq);
	}
	
	WordSet getWordSet(){
		return words;
	}
	
	int size() {
		return words.size();
	}
	
	double getDistanceToCentroid(){
		return distanceToCentroid;
	}
	
	void setDistanceToCentroid(double dist){
		distanceToCentroid = dist;
	}
	
}

/**
 * WordSet contains the words that are stored in each document, along with their frequencies.
 * 
 * @author Damien Beard
 *
 */
class WordSet {
	//The main hashmap is made up of <WordIndex, Count>
	HashMap<Integer,Double> words;
	long normalisationFactor;
	
	WordSet() {
		words = new HashMap<Integer,Double>();
	}
	
	void put(int word, int freq){
		words.put(word, (double)freq);
		normalisationFactor += freq*freq;
	}
	
	void normalise(){
		for(int word : words.keySet()) {
			words.put(word, words.get(word)/Math.sqrt(normalisationFactor));
		}
	}
	
	double getNormalisedFreq(int word){
		if (words.containsKey(word)) {
			return words.get(word);
		} else {
			return 0;
		}
	}
	
	long getNormalisationFactor() {
		return normalisationFactor;
	}
	
	int size() {
		return words.size();
	}
	
	HashMap<Integer,Double> getAllWords(){
		return words;
	}
}

/**
 * This is like a document vector, but contains the average of all closest document vector values
 * @author Damien Beard
 *
 */
class Centroid {
	int ID;
	HashMap<Integer,Double> vector;
	HashMap<Integer,Document> allDocs;
	ArrayList<Integer> docsAssigned;
	long normalisationFactor;
	HashMap<String, Integer> dictionary;
	
	Centroid(int ID, WordSet words, HashMap<Integer,Document> allDocs, HashMap<String, Integer> dictionary) {
		this.ID = ID;
		this.allDocs = allDocs;
		this.dictionary = dictionary;
		docsAssigned = new ArrayList<Integer>();
		vector = new HashMap<Integer,Double>();
		normalisationFactor = words.getNormalisationFactor();
		
		for (Integer word : words.getAllWords().keySet()){
			vector.put(word, words.getNormalisedFreq(word));
		}
	}
	
	/**
	 * Gets the closest five documents to the Centroid
	 * @return stack of closest documents, with closest one on top
	 */
	Stack<Integer> getTopFive(){
		Stack<Integer> topFive = new Stack<Integer>();
		Stack<Integer> buffer = new Stack<Integer>();
		
		for(int i=0;i<docsAssigned.size();i++){
			//reset buffer for new Document
			buffer = new Stack<Integer>();
			
			//The first document always makes it on the stack
			if (topFive.size()==0) {
				topFive.push(docsAssigned.get(i));
			} else {
				/* 
				 * If the current document is greater than the smallest value on the stack
				 * then we add it to it's correct place on the stack, getting rid of the 
				 * smallest value
				 */
				if (allDocs.get(docsAssigned.get(i)).getDistanceToCentroid() > 
		            allDocs.get(topFive.peek()).getDistanceToCentroid()) {
					while (topFive.size()>0&&allDocs.get(docsAssigned.get(i)).getDistanceToCentroid() > 
					        allDocs.get(topFive.peek()).getDistanceToCentroid()) {
						buffer.push(topFive.pop());
					}
					topFive.push(docsAssigned.get(i));
					while (topFive.size()<5&&buffer.size()>0) {
						topFive.push(buffer.pop());
					}
				}
			}
		}
		
		
		buffer = new Stack<Integer>();
		while (topFive.size()>0) {
			buffer.push(topFive.pop());
		}
		// Return stack from biggest -> smallest
		return buffer;
		
	}
	
	int size() {
		return vector.size();
	}
	
	void resetDocs() {
		docsAssigned = new ArrayList<Integer>();
	}
	
	long getNormalisationFactor() {
		return normalisationFactor;
	}
	
	HashMap<Integer,Double> getAllWords() {
		return vector;
	}
	
	void addDocs(Integer docID) {
		docsAssigned.add(docID);
	}
	
	void recalculateCentroid() {
		vector = new HashMap<Integer,Double>();
		for (String w : dictionary.keySet()) {
			Integer word = dictionary.get(w);
			double totalOccurence = 0.0;
			for (Integer docID : docsAssigned) {
				totalOccurence += allDocs.get(docID).getWordSet().getNormalisedFreq(word);
			}
			normalisationFactor += totalOccurence*totalOccurence;
			vector.put(word, totalOccurence/(double)docsAssigned.size());
		}
	}
}
/**
 * The Extractor class takes the words out of a file 'f', and returns it in a 
 * 'word' vs 'count' HashMap, to be processed by the K-means algorithm. Note that
 * this code includes extracts from Scott Sanner's Unigram Builder
 * @author Damo (with code form Scott Sanner's unigram builder_
 *
 */
class Extractor{
	StopWordChecker _swc = new StopWordChecker();
	TreeSet<WordCount> _wordCounts;
	HashMap<String,Integer> _topWord2Index;
	int num_top_words; boolean ignore_stop_words; boolean debug;
	
	Extractor(int num_top_words, boolean ignore_stop_words, boolean debug) {
		this.debug = debug;
		this.num_top_words = num_top_words;
		this.ignore_stop_words = ignore_stop_words;
	}
	
	/**
	 * Return a map of each word to its frequency in document f.
	 * @param f
	 * @return Hashmap of words -> frequencies in document f
	 */
	HashMap<String,Integer> getWordsAndCounts(File f) {
		HashMap<String,WordCount> word2count = new HashMap<String,WordCount>();
		String file_content = DocUtils.ReadFile(f);
		Map<Object,Double> word_counts = DocUtils.ConvertToFeatureMap(file_content);
		for (Map.Entry<Object, Double> me : word_counts.entrySet()) {
			String key = (String)me.getKey();
			WordCount wc = word2count.get(key);
			if (wc == null) {
				wc = new WordCount(key, (int)me.getValue().doubleValue());
				word2count.put(key, wc);
			} else
				wc._count++;
		}
		if (debug) System.out.println("Extracted " + word2count.size() + " unique tokens.");
		
		HashMap<String,Integer> wordsAndCounts = new HashMap<String,Integer>();
		int wordCount = 0;
		for (String key : word2count.keySet()) {
			if (ignore_stop_words && !_swc.isStopWord(word2count.get(key)._word)) {
				wordsAndCounts.put(word2count.get(key)._word, word2count.get(key)._count);
				//System.out.println("[index:" + wordCount + "] " + word2count.get(key)._word);
				wordCount++;
			}
			
			if (wordCount>num_top_words) break;
		}
		
		return wordsAndCounts;
	}
}