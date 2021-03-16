import os
import numpy as np
import json
import matplotlib.pyplot as plt

def ind2sub(array_shape, ind):
    rows = int((ind / array_shape[1]))
    cols = int((ind % array_shape[1]))
    return (rows, cols)

folder = 'almost_perfect'
root = './results/{}/'.format(folder)
logs = root + 'logs/'

epochs = 50
runs = 50

train_loss = np.zeros((runs,epochs))
train_acc = np.zeros((runs,epochs))
val_loss = np.zeros((runs,epochs))
val_acc = np.zeros((runs,epochs))
cnfs = np.zeros((runs,2,2))
cnf = np.zeros((2,2))

f = 0
k = np.zeros((5,), dtype=int)
for file in os.listdir(logs):
    if file.startswith('train_loss'):
        train_loss[k[0],:] = np.load(os.path.join(logs, file))
        k[0] += 1
    elif file.startswith('val_loss'):
        val_loss[k[1],:] = np.load(os.path.join(logs, file))
        k[1] += 1
    elif file.startswith('train_acc'):
        train_acc[k[2],:] = np.load(os.path.join(logs, file))
        k[2] += 1
    elif file.startswith('val_acc'):
        val_acc[k[3],:] = np.load(os.path.join(logs, file))
        k[3] += 1
    elif file.startswith('confusion'):
        cnf += np.load(os.path.join(logs, file))
        cnfs[k[4],:] = np.load(os.path.join(logs, file))
        k[4] += 1
    # if k[3] == 29:
    #     print(file)
# best model -- 37674
    
conf = os.path.join(root, 'configs/params.json')
with open(conf, 'r') as f:
    configs = json.load(f)
print(json.dumps(configs, indent=4))

print(np.max(train_acc, axis=1))
# print(np.min(train_loss, axis=1))
best_tr_acc = np.max(train_acc, axis=1)
print('Best training accuracy {:.2f} +- {:.4f}'.format(np.mean(best_tr_acc), np.std(best_tr_acc)))
print(np.max(val_acc, axis=1))
# print(np.min(val_loss, axis=1))
best_val_acc = np.max(val_acc, axis=1)
print('Best validation accuracy {:.2f} +- {:.4f}'.format(np.mean(best_val_acc), np.std(best_val_acc)))
print('Total confusion matrix \n{}'.format(cnf))

avg_tr_loss = np.mean(train_loss, axis=0)
avg_tr_acc = np.mean(train_acc, axis=0)
avg_val_loss = np.mean(val_loss, axis=0)
avg_val_acc = np.mean(val_acc, axis=0)

plt.figure()
plt.plot(avg_tr_loss, label='trian')
plt.plot(avg_val_loss, label='validation')
plt.legend(['train','validation'])

plt.figure()
plt.plot(avg_tr_acc, label='trian')
plt.plot(avg_val_acc, label='validation')
plt.legend(['train','validation'])