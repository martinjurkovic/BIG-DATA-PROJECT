import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def save_confusion_matrix(y_test, y_pred, model_name):
    cmat = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cmat, interpolation="nearest", cmap=plt.cm.Blues)
    for i in range(cmat.shape[0]):
        for j in range(cmat.shape[1]):
            ax.text(j, i, cmat[i, j], ha="center", va="center")
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
    plt.savefig(f"figs/confusion_matrix_{model_name}.png")
