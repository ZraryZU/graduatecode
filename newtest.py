import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torchsummary import summary
device = 'cuda' if torch.cuda.is_available() else 'cpu'

X_train = torch.tensor([[[[1,2,3,4],[2,3,4,5],[5,6,7,8],[1,3,4,5]]],[[[-1,2,3,-4],[2,-3,4,5],[-5,6,-7,8],[-1,-3,-4,-5]]]]).to(device).float()
X_train /= 8
y_train = torch.tensor([[0],[1]]).to(device).float()

def get_model():
    model = nn.Sequential(
        nn.Conv2d(1, 1, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(1, 1),
        nn.Sigmoid(),
    ).to(device)
    loss_fn = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-2)
    return model, loss_fn, optimizer


model, loss_fn, optimizer = get_model()
print(summary(model, tuple(X_train.shape[1:])))

def train_batch(x, y, model, optimizer, loss_fn):
    model.train()
    prediction = model(x)
    # print(prediction)
    batch_loss = loss_fn(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

trn_dl = DataLoader(TensorDataset(X_train, y_train))

for epoch in range(2000):
    for ix, batch in enumerate(iter(trn_dl)):
        x, y = batch
        batch_loss = train_batch(x, y, model, optimizer, loss_fn)
        print("Epoch"+str(epoch)+" Batch"+str(ix)+" "+str(batch_loss))

print(model(X_train[:1]))
# tensor([[0.0028]], device='cuda:0', grad_fn=<SigmoidBackward>)


print(list(model.children()))
"""
[Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1)), MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False), ReLU(), Flatten(start_dim=1, end_dim=-1), Linear(in_features=1, out_features=1, bias=True), Sigmoid()]
"""
