import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
import librosa.display as display

import sklearn.model_selection as ms

torch.manual_seed(1234)

dataset = []

use_gpu = torch.cuda.is_available()
print("GPU Available:{}".format(use_gpu))


def readFile(filepath):
    y, sr = librosa.load(filepath)

    D = librosa.stft(y)

    D_real, D_imag = np.real(D), np.imag(D)
    D_energy = np.sqrt(D_real**2 + D_imag**2)
    norm = librosa.util.normalize(D_energy)
    display.specshow(norm, y_axis="log", x_axis="time")
    result = np.pad(norm, ([(0, 0), (0, 315 - len(norm[0]))]), "constant")
    return result


def import_data(folder, i):
    for file in os.listdir(folder):
        try:
            temp = []
            f = folder + "/" + file
            temp.append(torch.tensor(readFile(f)))
            temp.append(torch.tensor(i))
            dataset.append(temp)
        except Exception:
            continue


"""Import Data"""

import_data("data/silence", 0)
import_data("data/noise", 1)
import_data("data/laugh", 2)
import_data("data/cry", 3)

train_set, test_set = ms.train_test_split(dataset, train_size=344)

BATCH_SIZE = 1

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True
)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)


class CNN_Baby(nn.Module):
    def __init__(self):
        super(CNN_Baby, self).__init__()
        self.fc1 = nn.Linear(1025 * 315, 4)
        self.fc3 = nn.Linear(500, 4)

    def forward(self, x):
        out = self.fc1(x)
        return out


cnn = CNN_Baby()
if use_gpu:
    cnn.cuda()

cnn.eval()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)


def train_baby(epoch, model, train_loader, optimizer):
    model.train()

    total_loss = 0
    correct = 0

    for i, (image, label) in enumerate(train_loader):

        optimizer.zero_grad()

        image = image.view(-1, 1025 * 315)

        if use_gpu:
            image = image.cuda()
            label = label.cuda()

        prediction = model(image)
        label = label.long()
        loss = criterion(prediction, label)

        loss.backward()

        optimizer.step()

        total_loss += loss
        pred_classes = prediction.data.max(1, keepdim=True)[1]
        correct += pred_classes.eq(label.data.view_as(pred_classes)).sum().double()

    mean_loss = total_loss / len(train_loader.dataset)
    acc = correct / len(train_loader.dataset)

    print(
        "Train Epoch: {}   Avg_Loss: {:.5f}   Acc: {}/{} ({:.3f}%)".format(
            epoch, mean_loss, correct, len(train_loader.dataset), 100.0 * acc
        )
    )

    return mean_loss, acc


def eval_baby(model, eval_loader):

    model.eval()

    total_loss = 0
    correct = 0

    for i, (image, label) in enumerate(eval_loader):

        optimizer.zero_grad()

        image = image.view(-1, 1025 * 315)

        if use_gpu:
            image = image.cuda()
            label = label.cuda()

        prediction = model(image)
        label = label.long()

        loss = criterion(prediction, label)

        loss.backward()

        optimizer.step()

        total_loss += loss

        pred_classes = prediction.data.max(1, keepdim=True)[1]

        correct += pred_classes.eq(label.data.view_as(pred_classes)).sum().double()

    mean_loss = total_loss / len(eval_loader.dataset)
    acc = correct / len(eval_loader.dataset)

    print(
        "Eval:  Avg_Loss: {:.5f}   Acc: {}/{} ({:.3f}%)".format(
            mean_loss, correct, len(eval_loader.dataset), 100.0 * acc
        )
    )

    return mean_loss, acc


def save_model(epoch, model, path="./"):

    filename = path + "neural_network_{}.pt".format(epoch)

    torch.save(model.state_dict(), filename)

    return model


def load_model(epoch, model, path="./"):

    filename = path + "neural_network_{}.pt".format(epoch)

    model.load_state_dict(torch.load(filename))

    return model


numEpochs = 20

checkpoint_freq = 10

path = "./"

train_losses = []
test_losses = []

train_accuracies = []
test_accuracies = []

for epoch in range(1, numEpochs + 1):

    train_loss, train_acc = train_baby(epoch, cnn, train_loader, optimizer)

    test_loss, test_acc = eval_baby(cnn, test_loader)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

    if epoch % checkpoint_freq == 0:
        save_model(epoch, cnn, path)

save_model(numEpochs, cnn, path)

print("\n\n\nOptimization ended.\n")


def get_label_index(tensor_pred):
    npray = tensor_pred.detach().numpy()
    index = [npray[0]]
    for x in range(0, len(npray) - 1):
        npray[x + 1] > npray[x]
        index = x + 1
    return index


results = []

f = readFile("cry.3gp")
p = cnn(torch.tensor(f).contiguous().view(-1, 1025 * 315))
print("Crying file test:{}".format(p))

results.append(get_label_index(p))

f = readFile("laugh_1.m4a_0.wav")
p = cnn(torch.tensor(f).contiguous().view(-1, 1025 * 315))
print("Laughing file test:{}".format(p))

results.append(get_label_index(p))

f = readFile("noise1.ogg")
p = cnn(torch.tensor(f).contiguous().view(-1, 1025 * 315))
print("Noise file test:{}".format(p))

results.append(get_label_index(p))

f = readFile("silence.wav_0.wav")
p = cnn(torch.tensor(f).contiguous().view(-1, 1025 * 315))
print("Silence file test:{}".format(p))

results.append(get_label_index(p))


def get_res():
    return np.asarray(results)
