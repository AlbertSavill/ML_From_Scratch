#include<iostream>
#include<vector>
#include<cmath>
#include<cstdlib>

// Data Point Structure
struct DataPoint {
	std::vector<double> features; // Feature Vector (x_i)
	int label; // CLass label, +1 or -1 (y_i)
};

// SVM Class 
class SVM {
private:
	// Function to compute the dot product of two vectors (w, x_i)
	double dot(const std::vector<double>& v1, const std::vector<double>& v2) {
		double result = 0;
		for (int i = 0; i < v1.size(); ++i) {
			result += v1[i] * v2[i]; // w * x_i
		}
		return result; // Returns the scalar result of dot product
	}

	double learningRate; // Learning Rate (eta)
	double regularizationParam; // Regularization parameter (C)
	int numIterations; // NUumber of iterations for gradient descent
	std::vector<double> weights; // Weight vecotr (w)
	double bias; // BIas term (b)

public:
	// Constructor to initialize the learning rate, regularization parameter (C), and # of iterations
	SVM(double learningRate, double regularizationParam, int numIterations)
	: learningRate(learningRate), regularizationParam(regularizationParam), numIterations(numIterations), bias(0) {}

	// Training the SVM using gradient descnet
	void train(const std::vector<DataPoint>& trainingData) {
		int n = trainingData.size();	// Number of data points
		int featureDim = trainingData[0].features.size(); // Dimension of feature space (d)

		// Initialize weight vector w to zero (initial guess)
		weights = std::vector<double>(featureDim, 0);

		// Gradient descent loop for numIterations times 
		for (int iter = 0; iter < numIterations; ++iter) {
			// Iterate through all data points in the training set
			for (const auto& DataPoint : trainingData) {
				// Compute dot product: w . x_i (the decision function)
				double DotProduct = dot(weights, DataPoint.features);

				// Check the classification condition: y_i * (w . x_i + b) THIS IS double condition
				// THe margin constraint: y_i * (w . x_i + b) >= 1
				double condition = DataPoint.label * (DotProduct + bias);

				// If the point is correctly classified or on the margin
				if (condition >= 1) {
					// No hinge loss for correctly classificed points
					// Update weights using regularization term only: w_j = w_j - eta * (2 * C * w_j)
					for (int j = 0; j < featureDim; ++j) {
						weights[j] -= learningRate * (2 * regularizationParam * weights[j]);
					}
				}
				// If the point is misclassified or within the margin
				else {
					// Apply both hinge loss and regularization to update weights
					// Update: w_j = w_j - eta * (2 * C * w_j - y_i * x_j)
					for (int j = 0; j < featureDim; ++j) {
						weights[j] -= learningRate * (2 * regularizationParam * weights[j] - DataPoint.label * DataPoint.features[j]);
					}

					// UPdate bias: b = b - Learning Rate (η) * (-y_i)
					bias -= learningRate * (-DataPoint.label);
				}
			}
		}
	}

	// Predict the label of a new data point (x_new)
	// Use the deciion function: f(x) = w . w + b, and return sign(f(x))
	int predict(const std::vector<double>& features) {
		double DotProduct = dot(weights, features);
		return (DotProduct + bias >= 0) ? 1 : -1; // If f(x) >= 0, reutrn 1, else -1
	}

	// Print the learned weights and bias (model parameters)
	void printModel() {
		std::cout << "Weights: ";
		for (const auto& w : weights) {
			std::cout << w << " ";
		}
		std::cout << "\nBias: " << bias << std::endl;
	}
};

int main() {
	// Sample training data (2D feature space)
	std::vector<DataPoint> trainingData = {
		{{2.0, 3.0}, 1},
		{{1.0, 1.0}, -1},
		{{2.0, 1.0}, -1},
		{{3.0, 3.0}, 1},
		{{5.0, 5.0}, 1},
		{{2.0, 0.5}, -1}
	};

	// Initialize the SVM with learning rate, regularization parameters (C), and number of iterations
	double learningRate = 0.00;
	double regularizationParam = 1.0;
	int numIterations = 1000;

	SVM svm(learningRate, regularizationParam, numIterations);

	// Train the SVM on the traiing data
	svm.train(trainingData);

	// Print the learned model parameters (weights and bias)
	svm.printModel();

	// Test on a new data point
	std::vector<double> testPoint = {4.0, 4.0}; // Test point with features [4.0, 4.0]
	int predictLabel = svm.predict(testPoint); // Predict hte label using the SVM model

	std::cout << "Predict label for test point: " << predictLabel << std::endl;

	return 0;
}