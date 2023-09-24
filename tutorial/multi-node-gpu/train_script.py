import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import horovod.torch as hvd


hvd.init()
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.set_device(hvd.local_rank())


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset,
    num_replicas=hvd.size(),
    rank=hvd.rank()
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    sampler=train_sampler
)

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.fc(x)

model = SimpleNet()
if torch.cuda.is_available():
    model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = hvd.DistributedOptimizer(
    optimizer,
    named_parameters=model.named_parameters()
)

for epoch in range(50):
    for i, (inputs, labels) in enumerate(train_loader):
        if torch.cuda.is_available():
            inputs, labels = inputs.cuda(), labels.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
