#include <iostream>
#include <stdio.h>
#include <vector>
#include <algorithm>


using namespace std;

bool custom_sort(double a, double b) // function sorts based on the minimum absolute value
{
	double a1 = abs(a - 0);
	double b1 = abs(b - 0);
	return a1 < b1;
}

int main()
{
	double x[] = { 1, 2, 3, 4, 5 };
	double y[] = { 1, 3, 3, 2, 5 };
	vector<double>error;				// array to store all error values
	double devi;
	double b0 = 0;
	double b1 = 0;
	double learnRate = 0.01;

	// Train Phase
	// Gradient descent algorithm y = mX+b 

	for (int i = 0; i < 20; i++) {
		int index = i % 5;				// access the index after each epoch
		double p = b0 + b1 * x[index];  // calculate prediction
		devi = p - y[index];			// calculating error
		b0 = b0 - learnRate * devi;		// updating b0
		b1 = b1 - learnRate * devi * x[index]; // updating b1
		cout << "B0 = " << b0 << "B1 = " << b1 << "Error = " << devi << endl; // print values after each update
		error.push_back(devi);
	}
	
	sort(error.begin(), error.end(), custom_sort);	// error values used to sort the data
	cout << "Optimal end values are: " << "b0 = " << b0 << " " << "b1 = " << b1 << "Error = " << error[0] << endl;

	// Testing Phase
	cout << "Enter a test x value: ";
	double test;
	cin >> test;
	double pred = b0 + b1 * test;
	cout << endl;
	cout << "The value predicted by the model = " << pred << endl;

	return 0;

}