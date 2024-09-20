#include<iostream>
#include<vector>
#include<cmath>
#include<algorithm>
#include<map>
#include<iomanip>

// Structure to represent a data point with features and a label

struct DataPoint {
	std::vector<double> features; // Feature vecotr (x_1, x_2, .... x_d)
	int label;					  // Class label eg 0 or 1 for binary classification
};

// FUnction to calculate the Euclidean distance between two data points
double euclideanDistance(const std::vector<double>& a, const std::vector<double>& b) {
	double sum = 0.0;
	// Ensure both vectors have the same dimension
	if (a.size() != b.size()) {
		std::cerr << "Error: Vectors have different dimensions!" << std::endl;
		return -1.0;
	}
	// Compute the sum of squared differences
	for (size_t i = 0; i < a.size(); ++i) {
		double diff = a[i] - b[i];
		sum += diff * diff;
	}
	// Take the square root to get Euclidean distancce
	return std::sqrt(sum);
}

// K-Nearest Neighbours (KNN) Classifier
class KNN {
private:
	int K; // Number of neighbours to consider
	std::vector<DataPoint> trainingData; // Training data

	// Function to get the K nearest neighbours of a query point
	std::vector<DataPoint> getNeighbours(const DataPoint& queryPoint, int K) const {
		// Vector to store pairs of (distance, DataPoint)
		std::vector<std::pair<double, DataPoint>> distanceDataPairs;

		// Calculate the distance from queryPoint to each training data point
		for (const auto& DataPoint : trainingData) {
			double distance = euclideanDistance(queryPoint.features, DataPoint.features);
			distanceDataPairs.emplace_back(distance, DataPoint);
		}

		// Sort the data points based on distance
		std::sort(distanceDataPairs.begin(), distanceDataPairs.end(),
			[](const std::pair<double, DataPoint>& a, const std::pair<double, DataPoint>& b) -> bool {
				return a.first < b.first;
			});

		// Extract the top K nearest neighbours
		std::vector<DataPoint> neighbours;
		for (int i = 0; i < K && i < distanceDataPairs.size(); ++i) {
			neighbours.push_back(distanceDataPairs[i].second);
		}

		return neighbours;
	}

	// Function to perform majority voting among neighbours
	int majorityVote(const std::vector<DataPoint>& neighbours) const {
		std::map<int, int> labelCounts; // Map to count occurrences of each label

		// Count the frequency of each label in the neighbours
		for (const auto& neighbour : neighbours) {
			labelCounts[neighbour.label]++;
		}

		// Find the label with the highest count
		int majorityLabel = -1;
		int maxCount = -1;
		for (const auto& pair : labelCounts) {
			if (pair.second > maxCount) {
				maxCount = pair.second;
				majorityLabel = pair.first;
			}
		}

		return majorityLabel;
	}

public:
	// Constructor to initialize K
	KNN(int K) : K(K) {}

	// Train function: for KNN, training is simply storing the training data
	void train(const std::vector<DataPoint>& trainingData) {
		this->trainingData = trainingData;
	}

	// Predict the label for a new data point
	int predict(const DataPoint& queryPoint) const {
		// Find the K nearest neighbours
		std::vector<DataPoint> neighbours = getNeighbours(queryPoint, K);
		// Perform majority vote among the neighbours
		return majorityVote(neighbours);
	}

};

int main() {
	// Example dataset
	// Features are 2-dimensional for simplicity
	std::vector<DataPoint> trainingData = {
	    // Class 0
	    {{1.0, 2.1}, 0},
	    {{1.5, 1.8}, 0},
	    {{2.0, 2.0}, 0},
	    {{2.5, 2.2}, 0},
	    // Class 1
	    {{3.0, 3.1}, 1},
	    {{3.5, 3.8}, 1},
	    {{4.0, 4.0}, 1},
	    {{4.5, 4.2}, 1}
	};

	// Create a KNN classifier with K = 3
	int K = 3;
	KNN knn(K);

	knn.train(trainingData);

	// Define some test data points
	std::vector<DataPoint> testData = {
		{{1.8, 1.9}, -1}, // Unknown label (to be predicted)
        {{3.2, 3.0}, -1},
        {{2.5, 2.5}, -1},
        {{4.1, 4.1}, -1}
	};

	// Predict the labels for the test data points
	std::cout << "KNN Classification Results (K = " << K << "):\n";
	for (size_t i = 0; i < testData.size(); ++i) {
		int predictedLabel = knn.predict(testData[i]);
		std::cout << "Test Point " << i + 1 << " ("
				  << testData[i].features[0] << ", "
				  << testData[i].features[1] << ") --> Predicted Label: "
				  << predictedLabel << std::endl;
	}

	return 0;
}