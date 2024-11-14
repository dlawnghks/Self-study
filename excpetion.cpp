#include <iostream>
#include <stdexcept>
using namespace std;

// 사용자 정의 예외 클래스
class DivisionByZero : public runtime_error {
public:
    DivisionByZero() : runtime_error("Custom Exception: Division by zero!") {}
};

class NegativeNumber : public invalid_argument {
public:
    NegativeNumber() : invalid_argument("Custom Exception: Negative numbers are not allowed!") {}
};

// 함수 : 나눗셈 수행 및 예외 발생
int divide(int numerator, int denominator) {
    if (denominator == 0) {
        throw DivisionByZero();
    }
    if (numerator < 0 || denominator < 0) {
        throw NegativeNumber();
    }

    // 오버플로우 예외 시뮬레이션 (임의 조건)
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
    // 사용자 정의 예외 처리
    catch (const DivisionByZero& e) {
        cerr << "Error: " << e.what() << endl;
    }
    catch (const NegativeNumber& e) {
        cerr << "Error: " << e.what() << endl;
    }
    // 표준 라이브러리 예외 처리
    catch (const invalid_argument& e) {
        cerr << "Invalid Argument Error: " << e.what() << endl;
    }
    catch (const overflow_error& e) {
        cerr << "Overflow Error: " << e.what() << endl;
    }
    // 기타 모든 예외 처리
    catch (...) {
        cerr << "An unknown error occurred." << endl;
    }

    return 0;
}
//예외처리를 사용할때는 cout 대신 cerr 출력문을 사용 
//이유는 cout은 일반적인 출력을 위해 사용하고 cerr는 에러 메시지 출력에 주로 사용. 이렇게 구분하면 에러와 정상 출력이 명확하게 구분되어 디버깅 시 더욱 직관적
