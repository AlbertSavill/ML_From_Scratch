/*
Linear Regression model by scratch in C++
Created by Albert Savill 18/09/2024 13:21
*/

#include<iostream>
#include<cmath>
#include<vector>

class LinearRegression {
private:
	double learningRate; // eta
	int numIterations; // n
	double w0, w1; // Parameters w0 = intercept, w1 = slope

	// Compute the cost (Mean Squared Error) for the current parameters
	double computeCost(const std::vector<double>& x, const std::vector<double>& y) const {
		double cost = 0;
		int n = x.size();
		for (int i = 0; i < n; ++i) {
			double y_hat = w0 + w1 * x[i]; // Predicted value
			double error = y[i] - y_hat; // Residual (error)
			cost += error * error; // Squared error
		}
		return cost / n; // MSE
	}

public:
	LinearRegression(double learningRate, int numIterations)
	: learningRate(learningRate), numIterations(numIterations), w0(0), w1(0) {}

	// Function to train the model using gradient descent
	void train(const std::vector<double>& x, const std::vector<double>& y) {
		int n = x.size(); // Number of data points

		// Gradient descent loop
		for (int iter = 0; iter < numIterations; ++iter) {
			double gradient_w0 = 0; // Gradient for intercept w0
			double gradient_w1 = 0; // Gradient for slope w1

			// Compute the gradient of the cost function with respect to w0 and w1
			for (int i = 0; i < n; ++i) {
				// Predicted value: y_hat = w0 + w1 * x_i
				double y_hat = w0 + w1 * x[i];

				// Compute residual (error): y_i - y_hat
				double error = y[i] - y_hat;

				// Update Gradient (partial derivatives of the cost function)
				// Partial derivative of cost w.r.t. w0: -(2/n) * sum(y_i - y_hat)
				gradient_w0 += -2 * error / n;

				// Partial derivative of cost w.r.t w1: -(2/n) * sum((y_i - y_hat) * x_i)
				gradient_w1 += -2 * error * x[i] / n;
			}

			// Update the parameters using the gradients and learning rate
			w0 -= learningRate * gradient_w0; // Update intercept
			w1 -= learningRate * gradient_w1; // Update slope

			// Print the cost (MSE) at every step for debugging if needed
			if (iter % 100 == 0) {
				double cost = computeCost(x, y);
				std::cout << "Iteration " << iter << ": Cost = " << cost << std::endl;
			}
		}
	}

	// Function to make predictions on new data
	double predict(double x) const {
		return w0 + w1 * x; // Prediction: y_hat = w0 + w1 * x
	}

	// Print the learned parameters (w0 and w1)
	void PrintModel() const {
		std::cout << "Model: y = " << w0 << " + " << w1 << " * x" << std::endl;
	}
};

int main() {
	// Sample dataset (x, y)
	std::vector<double> x = {1, 2, 3, 4, 5}; // Input feature
	std::vector<double> y = {2, 4, 5, 4, 5}; // Target output

	// Hyperparameters: learning rate and number of iterations
	double learningRate = 0.01;
	int numIterations = 1000;

	// Create and train the Linear Regression model
	LinearRegression model(learningRate, numIterations);
	model.train(x, y);

	// Print the learned model parameters
	model.PrintModel();

	// Test the model with a new input
	double testX = 6;
	double predictedY = model.predict(testX);
	std::cout << "Predicted value for x = " << testX << ": y = " << predictedY << std::endl;

	return 0;

}