import pandas as pd
import matplotlib.pyplot as plt

data = {
    "Model": ["Stefański", "LSTM", "Transformer", "CNN", "XGBoost", "LLM"],
    "Punch/No Punch": [82.99, 0, 98.9, 0, 0, 0],
    "All Classes":      [0, 12.5, 16.3, 78, 29, 12.08]
}

df = pd.DataFrame(data).set_index("Model")

fig, ax = plt.subplots(figsize=(10, 4))

df.plot(kind="bar", ax=ax)

ax.set_ylabel("F1 (%)", fontsize=18)
ax.set_xlabel("Model", fontsize=18)
ax.tick_params(axis="x", labelsize=18, rotation=0)
ax.tick_params(axis="y", labelsize=18)
ax.legend(fontsize=18)

plt.tight_layout()

plt.savefig("balanced_accuracy_grouped.svg", format="svg")