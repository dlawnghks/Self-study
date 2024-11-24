#include <iostream>
#include <fstream>
#include <stdexcept>
using namespace std;

// ����� ���� ���� Ŭ����
class DivisionByZero : public runtime_error {
public:
    DivisionByZero() : runtime_error("Custom Exception: Division by zero!") {}
};

class NegativeNumber : public invalid_argument {
public:
    NegativeNumber() : invalid_argument("Custom Exception: Negative numbers are not allowed!") {}
};

// ���� �α� ��� �Լ�
void logError(const string& message) {
    ofstream logFile("error_log.txt", ios::app);
    if (logFile.is_open()) {
        logFile << message << endl;
        logFile.close();
    }
}

// �Լ�: ����� �Է� ���� (������ �Է� �ޱ�)
int getValidatedInput(const string& prompt) {
    int value;
    while (true) {
        cout << prompt;
        cin >> value;

        if (cin.fail()) {
            cin.clear(); // �Է� ��Ʈ���� �ʱ�ȭ
            cin.ignore(numeric_limits<streamsize>::max(), '\n'); // �߸��� �Է� ����
            cerr << "Invalid input. Please enter an integer." << endl;
            logError("Invalid input: Non-integer value entered.");
        }
        else {
            return value;
        }
    }
}

// �Լ� : ������ ���� �� ���� �߻�
int divide(int numerator, int denominator) {
    if (denominator == 0) {
        throw DivisionByZero();
    }
    if (numerator < 0 || denominator < 0) {
        throw NegativeNumber();
    }

    // �����÷ο� ���� �ùķ��̼� (���� ����)
    if (numerator > 1000) {
        throw overflow_error("Overflow error: Numerator is too large!");
    }
    return numerator / denominator;
}

int main() {
    while (true) {
        try {
            // ����ڷκ��� ���� �Է� �ޱ�
            int num1 = getValidatedInput("Enter first number: ");
            int num2 = getValidatedInput("Enter second number: ");

            // ������ ����
            int result = divide(num1, num2);
            cout << "Result: " << result << endl;
        }
        // ����� ���� ���� ó��
        catch (const DivisionByZero& e) {
            cerr << "Error: " << e.what() << endl;
            logError(e.what());
        }
        catch (const NegativeNumber& e) {
            cerr << "Error: " << e.what() << endl;
            logError(e.what());
        }
        // ǥ�� ���̺귯�� ���� ó��
        catch (const invalid_argument& e) {
            cerr << "Invalid Argument Error: " << e.what() << endl;
            logError(e.what());
        }
        catch (const overflow_error& e) {
            cerr << "Overflow Error: " << e.what() << endl;
            logError(e.what());
        }
        // ��Ÿ ��� ���� ó��
        catch (const exception& e) {
            cerr << "Standard Exception: " << e.what() << endl;
            logError(e.what());
        }
        catch (...) {
            cerr << "An unknown error occurred." << endl;
            logError("Unknown error occurred.");
        }

        // ���α׷� ����� ���� Ȯ��
        char retry;
        cout << "Do you want to try again? (y/n): ";
        cin >> retry;
        if (retry != 'y' && retry != 'Y') break;
    }

    cout << "Program terminated. Goodbye!" << endl;
    return 0;
}
