import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Available devices ', torch.cuda.device_count())
print('Current cuda device ', torch.cuda.current_device())
print(torch.cuda.get_device_name(device))

# GPU 할당 변경하기
GPU_NUM = 0  # 원하는 GPU 번호 입력
device = torch.device(
    f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)  # change allocation of current GPU
print('Current cuda device ', torch.cuda.current_device())  # check

# Additional Infos
if device.type == 'cuda':
    print(torch.cuda.get_device_name(GPU_NUM))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(GPU_NUM)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(GPU_NUM)/1024**3, 1), 'GB')

# GPU 번호 확인


def get_available_gpu():
    gpu_list = []
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_list.append((i, gpu_name))
    return gpu_list


available_gpus = get_available_gpu()
print("사용 가능한 GPU:")
for gpu in available_gpus:
    print(f"GPU 번호: {gpu[0]}, 이름: {gpu[1]}")

# 로드 밸런싱을 위한 GPU 할당


def get_next_available_gpu():
    next_gpu_idx = torch.cuda.current_device() + 1
    if next_gpu_idx >= len(available_gpus):
        next_gpu_idx = 0
    return next_gpu_idx

# 예시를 위해 가상의 모델과 데이터를 사용합니다.


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)


model = Model()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 가상의 데이터셋을 생성합니다.
dataset = torch.randn(100, 10)
dataloader = DataLoader(dataset, batch_size=10)

# GPU에 모델을 올리고 로드 밸런싱을 통해 사용하지 않는 GPU를 할당합니다.
if torch.cuda.is_available():
    model = model.cuda()
    for batch in dataloader:
        inputs = batch.cuda()
        labels = torch.randn(10).cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 다음 배치에 대해 로드 밸런싱을 통해 GPU를 변경합니다.
        model = model.cuda(get_next_available_gpu())
