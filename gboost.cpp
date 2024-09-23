#include<iostream>
#include<vector>
#include<limits>
#include<cmath>
#include<algorithm>

// Decision Stump Class: A simple decision tree with depth of 1
class DecisionStump {
public:
	int feature_index; // Index of the feature to split on
	double threshold; // Threshold value for the split
	double left_value; // Prediction value for sample where features <= threshold
	double right_value; // Prediction value for the sample where features > threshold

	DecisionStump() : feature_index(-1), threshold(0.0), left_value(0.0), right_value(0.0) {}

	// Fit the decision stump to the data and residuals
	void fit(const std::vector<std::vector<double>>& X, const std::vector<double> &residuals) {
		int n_samples = X.size();
		if (n_samples == 0) return;
		int n_features = X[0].size();

		double min_error = std::numeric_limits<double>::max();

		// Iterate over all features to find the best split
		for (int feature = 0; feature < n_features; ++feature) {
			// Extract all unique value of the current feature
			std::vector<double> feature_values;
			for (const auto &sample : X) {
				feature_values.push_back(sample[feature]);
			}

			// Sort the feature values to consider possible thresholds
			std::vector<double> sorted_values = feature_values;
			std::sort(sorted_values.begin(), sorted_values.end());
			// Remove duplicates
			sorted_values.erase(std::unique(sorted_values.begin(), sorted_values.end(),
                [](double a, double b) { return std::abs(a - b) < 1e-9; }), sorted_values.end());

			// Try possible thresholds (midpoints between consecutive unique values)
			for (size_t i = 1; i < sorted_values.size(); ++i) {
				double current_threshold = (sorted_values[i - 1] + sorted_values[i]) / 2.0;

				// Split residuals into left and right based on the threshold
				double left_sum = 0.0, left_count = 0.0;
				double right_sum = 0.0, right_count = 0.0;

				for (int j = 0; j < n_samples; ++j) {
					if (X[j][feature] <= current_threshold) {
						left_sum += residuals[j];
						left_count += 1.0;
					} else {
						right_sum += residuals[j];
						right_count += 1.0;
					}
				}

				// Avoid division by zero
				if (left_count == 0 || right_count == 0) continue;

				// Compute the mean residuals for left and right
				double left_mean = left_sum / left_count;
				double right_mean = right_sum / right_count;

				// Compute the squared error
				double error = 0.0;
				for (int j = 0; j < n_samples; ++j) {
					double prediction = (X[j][feature] <= current_threshold) ? left_mean : right_mean;
					double diff = residuals[j] - prediction;
					error += diff * diff;
				}

				// Update the stump if a better split is found
				if (error < min_error) {
					min_error = error;
					feature_index = feature;
					threshold = current_threshold;
					left_value = left_mean;
					right_value = right_mean;
				}
			}
		}
	}

	// Predict the output for a single sample
	double predict(const std::vector<double> &x) const {
		if (x[feature_index] <= threshold) {
			return left_value;
		} else { 
			return right_value;
		}
	}
};

// Gradient Boosting class
class GradientBoostingRegressor {
public:
	int n_estimators; // Number of boosting iterations
	double learning_rate; // learning rate (nu)
	std::vector<DecisionStump> estimators; // List of weak learners
	double initial_prediction; // Initial prediction F0 

	GradientBoostingRegressor(int n_estimators_=100, double learning_rate_=0.1)
	 : n_estimators(n_estimators_), learning_rate(learning_rate_), initial_prediction(0.0) {}

	// Fit the model to the data
	void fit(const std::vector<std::vector<double>> &X, std::vector<double> &y) {
		int n_samples = X.size();
		if (n_samples == 0) return;
		int n_features = X[0].size();

		// Initialize F0 as the mean of y (for MSE loss)
		double sum = 0.0;
		for (double yi : y) sum += yi;
			initial_prediction = sum / n_samples;

		// Initialize predictions with F0
		std::vector<double> F(n_samples, initial_prediction);

		// Iteratively add estimators
		for (int m = 0; m < n_estimators; ++m) {
			// Compute residuals (negative gradient for MSE)
			std::vector<double> residuals(n_samples, 0.0);
			for (int i = 0; i < n_samples; ++i) {
				residuals[i] = y[i] = F[i];
			}

			// Fit a decision stump to the residuals
			DecisionStump stump;
			stump.fit(X, residuals);
			estimators.push_back(stump);

			// Update the predictions F(x) += learning_rate * h_m(x)
			for (int i = 0; i < n_samples; ++i) {
				F[i] += learning_rate * stump.predict(X[i]);
			}
		}
	}

	// Predict the output for new data
	double predict_single(const std::vector<double> &x) const {
		double prediction = initial_prediction;
		for (const auto &stump : estimators) {
			prediction += learning_rate * stump.predict(x);
		}
		return prediction;
	}

	// Predict for multiple samples
	std::vector<double> predict(const std::vector<std::vector<double>> &X) const {
		std::vector<double> predictions;
		for (const auto &x : X) {
			predictions.push_back(predict_single(x));
		}
		return predictions;
	}
};

int main() {
	// Example dataset: Simple linear relationships with noise
	std::vector<std::vector<double>> X = {
		{1.0}, {2.0}, {3.0}, {4.0}, {5.0},
        {6.0}, {7.0}, {8.0}, {9.0}, {10.0}
	};

	std::vector<double> y = {1.2, 1.9, 3.0, 3.8, 5.1, 5.9, 7.2, 8.0, 9.1, 10.2};

	// Initialize Gradient Boosting Regressor
	GradientBoostingRegressor gbr(100, 0.1);
	gbr.fit(X, y);

	// Make Predictions
	std::vector<std::vector<double>> X_test = {
		{1.5}, {2.5}, {3.5}, {4.5}, {5.5},
		{6.5}, {7.5}, {8.5}, {9.5}, {10.5}

	};
	std::vector<double> predictions = gbr.predict(X_test);

	// Output predictions
	std::cout << "Predictions:\n";
	for (size_t i = 0; i < X_test.size(); ++i) {
		std::cout << "x = " << X_test[i][0] << ", y_pred = " << predictions[i] << "\n";
	}

	return 0;

}