import torch
from torch import nn

# 학습 데이터
inputs = torch.Tensor([1, 2, 3, 4, 5, 6]).view(6, 1)
targets = torch.Tensor([3, 4, 5, 6, 7, 8]).view(6, 1)

class SimpleLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Sequential(
            nn.Linear(1, 1)  # 입력 데이터 1개, 출력 데이터 1개
        )

    def forward(self, x):
        output = self.linear_layer(x)
        return output

# 모델, 손실 함수, 옵티마이저 정의
model = SimpleLinearModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 학습 설정
num_epochs = 2000
for epoch in range(num_epochs + 1):
    predictions = model(inputs)
    loss = loss_fn(predictions, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Epoch:', epoch, ', Loss:', loss.item())

# 테스트 데이터
test_inputs = torch.Tensor([3.0, 1.0, 1.2, -2.5]).view(4, 1)
test_predictions = model(test_inputs)
print(test_predictions)
