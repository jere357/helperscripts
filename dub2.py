import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

class ConvolutionalModel(nn.Module):
    def __init__(self, batch_size, in_channels, conv1_num_filters, conv2_num_filters, fc1_width, class_count):
        super(ConvolutionalModel, self).__init__()
        self.batch_size = batch_size
        self.conv1 = nn.Conv2d(in_channels, conv1_num_filters, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2 = nn.Conv2d(conv1_num_filters, conv2_num_filters, kernel_size=5, stride=1, padding=2, bias=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv2_num_filters*7*7 , fc1_width, bias=True)
        self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)

        # parametri su već inicijalizirani pozivima Conv2d i Linear
        # ali možemo ih drugačije inicijalizirati
        #self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m is not self.fc_logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
        self.fc_logits.reset_parameters()

    def forward(self, x):
        h = self.conv1(x)
        h = self.maxpool(h)
        h = self.relu(h)
        h = self.conv2(h)
        h = self.maxpool(h)
        h = self.relu(h)
        h = self.flatten(h)
        #h = h.view(h.shape[0], -1)
        h = self.fc1(h)
        h = self.relu(h)
        logits = self.fc_logits(h)
        return logits

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X,y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 400 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    return loss

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X,y in dataloader:
            pred = model(X)
            loss = loss_fn(pred, y)
            test_loss += loss
            correct += (pred.argmax(1) == y ).type(torch.float).sum().item()
            pass
            
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct*100

if __name__ =="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    writer = SummaryWriter() #tensorboard writer
    config_dict = {}
    config_dict['batch_size'] = 64
    config_dict['num_epochs'] = 10
    print(f"Using {device} device")
    #LOAD THE DATAR
    training_data = datasets.MNIST(
        root='data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    test_data  = datasets.MNIST(
        root='data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)
    model = ConvolutionalModel(batch_size = 64, in_channels=1, conv1_num_filters=16, conv2_num_filters=32, fc1_width=512, class_count=10).to(device)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_loss_list = []
    test_loss_list = []
    for epoch in range(config_dict["num_epochs"]):
        print(f"Epoha: {epoch}")
        train_loss = train_loop(train_dataloader, model, loss, optimizer)
        test_loss, test_acc = test_loop(test_dataloader, model, loss)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/test", test_loss, epoch)
        writer.add_scalar("accuracy %/test", test_acc, epoch)
    writer.flush()


    #plt.plot(train_loss_list, label="train loss")
    #plt.plot(test_loss_list, label="test loss")
    #plt.show()
    writer.close()
