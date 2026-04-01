"""
Audio Classification — Comparative Evaluation
===============================================
Compares 4 classifier-filter combinations on audio data:
  - KNN (k=5) + Gabor filters
  - KNN (k=5) + Mexican Hat filters
  - Random Forest (100 trees) + Gabor filters
  - Random Forest (100 trees) + Mexican Hat filters

Generates comparison charts and confusion matrix for the best model.

Usage:
    Place data.mat in the project directory and run:
    $ python clasificare_completa.py
"""

import numpy as np
from get_features import get_features, get_features_mexican
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import scipy
import matplotlib.pyplot as plt



def main():
    data = scipy.io.loadmat('data.mat')
    audio_train, audio_test = data['audio_train'].T, data['audio_test'].T
    labels_train, labels_test = data['labels_train'], data['labels_test']
    fs = data['fs'][0, 0]

    alpha = 1.0
    start1 = audio_train.shape[1] // 2 - int(alpha * audio_train.shape[1] // 2) + 1
    end1 = audio_train.shape[1] // 2 + int(alpha * audio_train.shape[1] // 2)
    audio_train_small = audio_train[:, start1:end1]

    start2 = audio_test.shape[1] // 2 - int(alpha * audio_test.shape[1] // 2) + 1
    end2 = audio_test.shape[1] // 2 + int(alpha * audio_test.shape[1] // 2)
    audio_test_small = audio_test[:, start2:end2]

    labels_train = labels_train[:, 0]
    labels_test = labels_test[:, 0]

    feat_train_gabor = get_features(audio_train_small, fs)
    feat_test_gabor = get_features(audio_test_small, fs)

    feat_train_mexican = get_features_mexican(audio_train_small, fs)
    feat_test_mexican = get_features_mexican(audio_test_small, fs)

    results = {}
    predictions = {}

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(feat_train_gabor, labels_train)
    pred_test = clf.predict(feat_test_gabor)
    acc_test = np.mean(pred_test == labels_test)
    results['KNN + Gabor'] = acc_test
    predictions['KNN + Gabor'] = pred_test

    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(feat_train_mexican, labels_train)
    pred_test = clf.predict(feat_test_mexican)
    acc_test = np.mean(pred_test == labels_test)
    results['KNN + Mexican Hat'] = acc_test
    predictions['KNN + Mexican Hat'] = pred_test

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(feat_train_gabor, labels_train)
    pred_test = clf.predict(feat_test_gabor)
    acc_test = np.mean(pred_test == labels_test)
    results['RF + Gabor'] = acc_test
    predictions['RF + Gabor'] = pred_test

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(feat_train_mexican, labels_train)
    pred_test = clf.predict(feat_test_mexican)
    acc_test = np.mean(pred_test == labels_test)
    results['RF + Mexican Hat'] = acc_test
    predictions['RF + Mexican Hat'] = pred_test

    methods = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(12, 7))
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = plt.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    plt.ylabel('Accuracy', fontsize=13, fontweight='bold')
    plt.title('Comparison of Classification Methods', fontsize=15, fontweight='bold')
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height, f'{height*100:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig('RatAlexia-Maria_343C2_comparatie.png', dpi=300, bbox_inches='tight')
    plt.close()

    knn_results = [results['KNN + Gabor'], results['KNN + Mexican Hat']]
    rf_results = [results['RF + Gabor'], results['RF + Mexican Hat']]
    filters = ['Gabor', 'Mexican Hat']
    x = np.arange(len(filters))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, knn_results, width, label='KNN', color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, rf_results, width, label='Random Forest', color='#e74c3c', alpha=0.8, edgecolor='black')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Detailed Comparison: Classifiers × Filters', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(filters)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{height*100:.1f}%',
                   ha='center', va='bottom', fontsize=10)
    plt.tight_layout()
    plt.savefig('RatAlexia-Maria_343C2_detailed_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    best_method = max(results, key=results.get)
    best_accuracy = results[best_method]
    best_pred = predictions[best_method]
    cm = confusion_matrix(labels_test, best_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap='Blues', values_format='d', colorbar=True)
    plt.title(f'Confusion Matrix - {best_method}\nAccuracy: {best_accuracy*100:.2f}%',
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('RatAlexia-Maria_343C2_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Rezultate:")
    for method, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {method:25s}: {acc*100:6.2f}%")

    


if __name__ == "__main__":
    main()
