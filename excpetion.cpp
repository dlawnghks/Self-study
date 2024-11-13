#include <iostream>
using namespace std;

int divide(int numerator, int denominator) {
    if (denominator == 0) {
        throw runtime_error("Division by zero!");
    }
    return numerator / denominator;
}

int main() {
    int num1, num2, result;

    try {
        cout << "Enter first numbers: ";
        cin >> num1;
        cout << "Enter second numbers: ";
        cin >> num2;

        result = divide(num1, num2);
        cout << "Result: " << result << endl;
    }
    catch (const runtime_error& e) {
        cerr << "Error: " << e.what() << endl;
    }
    catch (...) {
        cerr << "An unknown error occurred." << endl;
    }
    return 0;
}