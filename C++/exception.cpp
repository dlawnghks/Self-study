#include <iostream>
#include <fstream>
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

// 에러 로그 기록 함수
void logError(const string& message) {
    ofstream logFile("error_log.txt", ios::app);
    if (logFile.is_open()) {
        logFile << message << endl;
        logFile.close();
    }
}

// 함수: 사용자 입력 검증 (정수만 입력 받기)
int getValidatedInput(const string& prompt) {
    int value;
    while (true) {
        cout << prompt;
        cin >> value;

        if (cin.fail()) {
            cin.clear(); // 입력 스트림을 초기화
            cin.ignore(numeric_limits<streamsize>::max(), '\n'); // 잘못된 입력 무시
            cerr << "Invalid input. Please enter an integer." << endl;
            logError("Invalid input: Non-integer value entered.");
        }
        else {
            return value;
        }
    }
}

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
        throw overflow_error("Overflow error: Numerator is too large!");
    }
    return numerator / denominator;
}

int main() {
    while (true) {
        try {
            // 사용자로부터 숫자 입력 받기
            int num1 = getValidatedInput("Enter first number: ");
            int num2 = getValidatedInput("Enter second number: ");

            // 나눗셈 수행
            int result = divide(num1, num2);
            cout << "Result: " << result << endl;
        }
        // 사용자 정의 예외 처리
        catch (const DivisionByZero& e) {
            cerr << "Error: " << e.what() << endl;
            logError(e.what());
        }
        catch (const NegativeNumber& e) {
            cerr << "Error: " << e.what() << endl;
            logError(e.what());
        }
        // 표준 라이브러리 예외 처리
        catch (const invalid_argument& e) {
            cerr << "Invalid Argument Error: " << e.what() << endl;
            logError(e.what());
        }
        catch (const overflow_error& e) {
            cerr << "Overflow Error: " << e.what() << endl;
            logError(e.what());
        }
        // 기타 모든 예외 처리
        catch (const exception& e) {
            cerr << "Standard Exception: " << e.what() << endl;
            logError(e.what());
        }
        catch (...) {
            cerr << "An unknown error occurred." << endl;
            logError("Unknown error occurred.");
        }

        // 프로그램 재시작 여부 확인
        char retry;
        cout << "Do you want to try again? (y/n): ";
        cin >> retry;
        if (retry != 'y' && retry != 'Y') break;
    }

    cout << "Program terminated. Goodbye!" << endl;
    return 0;
}
