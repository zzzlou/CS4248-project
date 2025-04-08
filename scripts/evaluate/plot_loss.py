import matplotlib.pyplot as plt

epochs = list(range(1, 21))
train_acc = [0.6792, 0.6813, 0.6819, 0.7312, 0.7597, 0.7889, 0.8257, 0.8583, 0.8715, 0.9049,
             0.9132, 0.9306, 0.9403, 0.9674, 0.9743, 0.9854, 0.9826, 0.9847, 0.9812, 0.9889]
val_acc = [0.6611, 0.6611, 0.6722, 0.7389, 0.7694, 0.7889, 0.8194, 0.8306, 0.8417, 0.8417,
           0.8444, 0.8667, 0.8694, 0.8667, 0.8778, 0.8611, 0.8833, 0.8333, 0.8500, 0.8722]
loss_cls = [113.9135, 109.1764, 105.4306, 100.3682, 94.0457, 85.3583, 75.3171, 64.1133, 56.8287, 44.5033,
            38.2933, 31.5045, 25.0806, 17.4439, 12.8052, 10.3310, 9.7646, 8.2249, 8.9116, 5.6877]
loss_type = [280.4368, 179.0824, 145.7054, 131.6528, 122.5481, 111.2984, 99.8190, 92.7897, 83.3257, 77.7636,
             72.1219, 69.5222, 60.1823, 57.4561, 56.8441, 54.8792, 54.4696, 53.7525, 54.5551, 49.5826]
loss_bal = [1089.1979, 1105.8142, 1115.2346, 1122.4091, 1137.3923, 1149.4216, 1163.9625, 1170.8226, 1194.6027,
            1236.1141, 1245.3545, 1229.2983, 1235.3053, 1243.4868, 1224.1358, 1205.7616, 1208.0060, 1182.6796,
            1172.6235, 1182.7888]

plt.figure(figsize=(12, 6))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, label='Train Accuracy')
plt.plot(epochs, val_acc, label='Val Accuracy')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.grid(True)

# Plot Losses
plt.subplot(1, 2, 2)
plt.plot(epochs, loss_cls, label='Cls Loss')
plt.plot(epochs, loss_type, label='Type Loss')
plt.plot(epochs, loss_bal, label='Balance Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Components over Epochs")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("loss_plot.png")
plt.show()
