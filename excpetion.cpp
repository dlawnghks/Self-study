#include <iostream>
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
        throw overflow_error("Overflow error: numerator is too large!");
    }
    return numerator / denominator;
}

int main() {
    int num1, num2, result;

    try {
        cout << "Enter first number: ";
        cin >> num1;
        cout << "Enter second number: ";
        cin >> num2;

        if (cin.fail()) {
            throw invalid_argument("Invalid input: Please enter integers only.");
        }

        result = divide(num1, num2);
        cout << "Result: " << result << endl;
    }
    // ����� ���� ���� ó��
    catch (const DivisionByZero& e) {
        cerr << "Error: " << e.what() << endl;
    }
    catch (const NegativeNumber& e) {
        cerr << "Error: " << e.what() << endl;
    }
    // ǥ�� ���̺귯�� ���� ó��
    catch (const invalid_argument& e) {
        cerr << "Invalid Argument Error: " << e.what() << endl;
    }
    catch (const overflow_error& e) {
        cerr << "Overflow Error: " << e.what() << endl;
    }
    // ��Ÿ ��� ���� ó��
    catch (...) {
        cerr << "An unknown error occurred." << endl;
    }

    return 0;
}
//����ó���� ����Ҷ��� cout ��� cerr ��¹��� ��� 
//������ cout�� �Ϲ����� ����� ���� ����ϰ� cerr�� ���� �޽��� ��¿� �ַ� ���. �̷��� �����ϸ� ������ ���� ����� ��Ȯ�ϰ� ���еǾ� ����� �� ���� ������
