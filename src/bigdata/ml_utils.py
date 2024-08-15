import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(y_test, y_pred, model_name):
    cmat = confusion_matrix(y_test, y_pred, normalize="all")
    fig, ax = plt.subplots()
    ax.imshow(cmat, interpolation="nearest", cmap=plt.cm.Blues)
    for i in range(cmat.shape[0]):
        for j in range(cmat.shape[1]):
            ax.text(j, i, f"{cmat[i, j]:.2f}", ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.axis("equal")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(
        [
            "Out of State",
            "NY",
        ]
    )
    ax.set_yticklabels(["Out of State", "NY"])
    fig.tight_layout()
    plt.savefig(f"figs/confusion_matrix_{model_name}.png", dpi=300)
    plt.close()


def plot_results(y_test, y_pred, model_name, fmt="csv", title=""):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([0, max_val], [0, max_val], color="red", linestyle="--")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(f"figs/results_{fmt}_{model_name}.png", dpi=300)
    plt.close()
