import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# 간단한 신경망 모델 정의
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 5)  # 입력 크기 2, 출력 크기 5
        self.fc2 = nn.Linear(5, 1)  # 입력 크기 5, 출력 크기 1

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# 데이터 준비 (x는 2개의 특성, y는 이진 레이블)
x_data = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_data = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# DataLoader로 배치 처리
dataset = TensorDataset(x_data, y_data)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# 모델, 손실 함수, 옵티마이저 정의
model = SimpleNN()
criterion = nn.MSELoss()  # 손실 함수 (회귀 문제)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 모델 훈련
epochs = 1000
for epoch in range(epochs):
    for inputs, labels in dataloader:
        # 순전파
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 역전파
        optimizer.zero_grad()  # 기울기 초기화
        loss.backward()  # 기울기 계산
        optimizer.step()  # 파라미터 업데이트

    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

# 모델 테스트
with torch.no_grad():
    test_input = torch.tensor([[0.5, 0.5]])
    test_output = model(test_input)
    print(f'Test input: {test_input}, Model output: {test_output.item():.4f}')
