import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns


#import dataset
iris = sns.load_dataset('iris')
# print(iris.head())
sns.pairplot(iris, hue=("species"))

data = torch.tensor(iris[iris.columns[:4]].values).float()
labels = torch.zeros(len(data), dtype=torch.long)
labels[iris.species == "versicolor"] = 1
labels[iris.species == "virginica"] = 2
# print(labels)
# =============================================================================
# ANN Model
# =============================================================================
ANNiris = nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 3),
)

# loss function
lossfun = nn.CrossEntropyLoss()
# optimizer
optimizer = torch.optim.SGD(ANNiris.parameters(), lr=0.1)


# training
numepoch = 1000
# initial loss
losses = torch.zeros(numepoch)
ongoingaccuracy = []
for epoch in range(numepoch):
    print(epoch)
    yhat = ANNiris(data)
    loss = lossfun(yhat, labels)
    losses[epoch] = loss

    # back prop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # accusracy
    matches = torch.argmax(yhat, axis=1) == labels
    matchesNumeric = matches.float()
    accuracyPct = 100*torch.mean(matchesNumeric)
    ongoingaccuracy.append(accuracyPct)


predictions = ANNiris(data)
predLabels = torch.argmax(predictions, axis=1)
totalacc = 100*torch.mean((predLabels == labels).float())
print(totalacc)

# =============================================================================
#     Visualize
# =============================================================================
# report accuracy
print('Final accuracy: %g%%' % totalacc)

fig, ax = plt.subplots(1, 2, figsize=(13, 4))

ax[0].plot(losses.detach())
ax[0].set_ylabel('Loss')
ax[0].set_xlabel('epoch')
ax[0].set_title('Losses')

ax[1].plot(ongoingaccuracy)
ax[1].set_ylabel('accuracy')
ax[1].set_xlabel('epoch')
ax[1].set_title('Accuracy')
plt.show()
# run training again to see whether this performance is consistent
sm = nn.Softmax(1)
torch.sum(sm(yhat), axis=1)
# plot the raw model outputs

fig = plt.figure(figsize=(10, 4))

plt.plot(sm(yhat.detach()), 's-', markerfacecolor='w')
plt.xlabel('Stimulus number')
plt.ylabel('Probability')
plt.legend(['setosa', 'versicolor', 'virginica'])
plt.show()

# try it again without the softmax!
