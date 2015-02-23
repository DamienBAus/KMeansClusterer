 This program implements the K-means algorithm. The variable parameters are given in the main
 function. The stopping criteria condition is that it has performed 100 iterations, or that
 the total number of documents assigned to each cluster remains constant for a set number of
 iterations.
 @author Damien Beard
 
Kmeans.java contains 2 variables in the main program, the number of 
clusters and the directory path for the test data. The K-means algorithm is
performed within the performKMeansAlgorithm(K, inputFilePath) method. 

The following classes are held within the file Kmeans.java:

The Document class contains an ID, a file name, the distance to its closest 
centroid, and a set of words that it contains (stored in a variable of type
WordSet).

The WordSet class is used to store words in a document, along with their 
normalised frequencies.

The Centroid class contains a vector that sits amongst a set of documents. 
As the name suggests, it acts as a centroid within the K-means algorithm.

The Extractor class takes the words out of a file 'f', and returns it in a 
'word' vs 'count' HashMap, to be processed by the K-means algorithm.
 
 To use this program, you should download the nlp.nicta.filters.StopWordChecker package, or 
 replace this package with another stop word checker. 