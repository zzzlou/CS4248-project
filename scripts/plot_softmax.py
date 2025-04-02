import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("router_softmax.csv")
df.plot(marker='o', figsize=(10, 5))
plt.xlabel("Epoch")
plt.ylabel("Average Softmax Weight")
plt.title("Router Softmax Weight per Expert across Epochs")
plt.grid(True)
plt.legend(title="Expert")
plt.tight_layout()
plt.show()