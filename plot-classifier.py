import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.nn.functional import softmax
from sklearn.metrics import (
    auc,
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve
)

CV = 5

plt.style.use('./misc/style.mplstyle')

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

for source, label in [
    ("models/LeNet5-LabFrame-Classifier-results.npy", "LeNet-LabFrame"),
    ("models/ResNet18-LabFrame-Classifier-results.npy", "ResNet18-LabFrame"),
    ("models/ResNet50-LabFrame-Classifier-results.npy", "ResNet50-LabFrame")]:
                      
    results = np.load(source)
    y_true = results[:, 0]
    y_pred = softmax(torch.tensor(results[:, 1:]), 1).numpy()[:, 1]


    print("Accuracy for {0}: {1:.2f}".format(
        label, 100*accuracy_score(y_true, y_pred > .5)
    ))

    samples_per_cv = len(y_true) // CV

    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 1000)
    for i in range(CV):
        y_true_cv = y_true[i*samples_per_cv:(i+1)*samples_per_cv]
        y_pred_cv = y_pred[i*samples_per_cv:(i+1)*samples_per_cv]

        fpr, tpr, _ = roc_curve(y_true_cv, y_pred_cv)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)


    ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=0.5)
    ax1.plot(mean_fpr, mean_tpr, label=r'[{0:.2f} $\pm {1:.2f}$ AUC] {2}'.format(
        mean_auc,
        std_auc,
        label
    ))

    p, r, _ = precision_recall_curve(y_true, y_pred)
    ax2.plot(r, p, label=r'[{0:.2f} AP] {1}'.format(
        average_precision_score(y_true, y_pred), label
    ))
    
ax1.set_xscale('log')
ax1.set_ylim([-0.05, 1.05])
ax1.set_xlabel("False Positive Rate", horizontalalignment='right', x=1.0)
ax1.set_ylabel("True Positive Rate", horizontalalignment='right', y=1.0)
ax1.legend(bbox_to_anchor=(1., 1.))
fig1.savefig('models/ROC', bbox_inches='tight')

ax2.set_xlim([-0.05, 1.05])
ax2.set_ylim([-0.05, 1.05])
ax2.set_xlabel("Recall", horizontalalignment='right', x=1.0)
ax2.set_ylabel("Precision", horizontalalignment='right', y=1.0)
ax2.legend(bbox_to_anchor=(1., 1.))
fig2.savefig('models/PR', bbox_inches='tight')
