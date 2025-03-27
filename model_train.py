import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from pathlib import Path
from timeit import default_timer as timer

device = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATH = Path("data/ovocko")
train_dir = DATA_PATH / "train"
test_dir = DATA_PATH / "test"

train_trans = transforms.Compose([transforms.Resize(size=(32, 32)),
                                  transforms.TrivialAugmentWide(num_magnitude_bins=20),
                                  transforms.ToTensor()])

test_trans = transforms.Compose([transforms.Resize(size=(32, 32)),
                                 transforms.ToTensor()])

train_data = datasets.ImageFolder(root=train_dir, transform=train_trans)
test_data = datasets.ImageFolder(root=test_dir, transform=test_trans)

BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)


class FruitImgClassification(nn.Module):
    def __init__(self, input_num, hidden_units, output_num):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_num, out_channels=hidden_units, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*5**2, out_features=output_num),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        return self.classifier_layer(self.block2(self.block1(x)))


classes = train_data.classes

model = FruitImgClassification(input_num=3, hidden_units=16, output_num=len(classes)).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)


def train_step(data_loader, model, loss_fn, optimizer, device=device):
    model.train()
    train_loss, train_acc = 0, 0
    for X, y in data_loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        labels = torch.argmax(logits, dim=1)
        loss = loss_fn(logits, y)
        train_loss += loss.item()
        acc = (labels == y).sum().item()/len(labels)
        train_acc += acc
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    return train_loss, train_acc


def test_step(data_loader, model, loss_fn, device=device):
    model.eval()
    test_loss, test_acc = 0, 0
    for X_test, y_test in data_loader:
        X_test, y_test = X_test.to(device), y_test.to(device)
        test_logits = model(X_test)
        test_labels = torch.argmax(test_logits, dim=1)
        loss = loss_fn(test_logits, y_test)
        test_loss += loss.item()
        acc = (test_labels == y_test).sum().item()/len(test_labels)
        test_acc += acc
    test_loss /= len(data_loader)
    test_acc /= len(data_loader)
    return test_loss, test_acc


def train_and_test(train_dataloader, test_dataloader, model, loss_fn, optimizer, epochs):
    train_loss, train_acc, test_loss, test_acc = [], [], [], []
    for epoch in range(epochs):
        train_loss_value, train_acc_value = train_step(data_loader=train_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer)
        test_loss_value, test_acc_value = test_step(data_loader=test_dataloader, model=model, loss_fn=loss_fn)
        train_loss.append(train_loss_value)
        train_acc.append(train_acc_value)
        test_loss.append(test_loss_value)
        test_acc.append(test_acc_value)
        print(f"Epoch: {epoch} | Train acc: {100*train_acc_value:.2f}% | Test acc: {100*test_acc_value:.2f}%")
    return train_loss, train_acc, test_loss, test_acc


EPOCHS = 20
start_time = timer()
train_loss, train_acc, test_loss, test_acc = train_and_test(train_dataloader=train_dataloader, test_dataloader=test_dataloader, model=model, loss_fn=loss_fn, optimizer=optimizer, epochs=EPOCHS)
end_time = timer()
print(f"Final time: {end_time-start_time}s")


def plot_data(train_loss=train_loss, train_acc=train_acc, test_loss=test_loss, test_acc=test_acc):
    epochs = range(len(train_loss))
    plt.figure(figsize=(10, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, test_loss, label="test loss")
    plt.title("LOSS")
    plt.xlabel("epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, label="train acc")
    plt.plot(epochs, test_acc, label="test acc")
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.legend()

    plt.show()


plot_data()
