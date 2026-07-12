# Evaluate baseline model versus hierarchical model
# ============================================================================
# FINAL EVALUATION
# ============================================================================

print(f"\n{'='*70}")
print("FINAL EVALUATION")
print(f"{'='*70}")

best_path = MODEL_BEST_PATH if os.path.exists(MODEL_BEST_PATH) else MODEL_LATEST_PATH
print(f"Loading best model from: {best_path}")
model = tf.keras.models.load_model(best_path)


def evaluate_split(dataset, split_name, n_samples):
    print(f"\n--- {split_name.upper()} SET ---")
    y_pred_probs = model.predict(dataset, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Collect true labels from dataset
    y_true = np.concatenate([
        np.argmax(y.numpy(), axis=1)
        for _, y in dataset
    ])[:n_samples]
    y_pred = y_pred[:n_samples]
    y_pred_probs = y_pred_probs[:n_samples]

    acc  = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0)
    avg_conf = np.mean(np.max(y_pred_probs, axis=1))

    y_true_cat = keras.utils.to_categorical(y_true, num_classes)
    top3 = keras.metrics.top_k_categorical_accuracy(
        y_true_cat, y_pred_probs, k=3).numpy().mean()
    top5 = keras.metrics.top_k_categorical_accuracy(
        y_true_cat, y_pred_probs, k=5).numpy().mean()

    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"  Top-3     : {top3*100:.2f}%")
    print(f"  Top-5     : {top5*100:.2f}%")
    print(f"  Avg Conf  : {avg_conf*100:.2f}%")

    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                top3=top3, top5=top5, avg_conf=avg_conf,
                y_true=y_true, y_pred=y_pred, y_pred_probs=y_pred_probs)


val_res  = evaluate_split(val_ds,  'Validation', n_val)
test_res = evaluate_split(test_ds, 'Test (held-out)', n_test)

hierarchical_acc = 98.40

print(f"\n{'='*70}")
print("COMPARISON")
print(f"{'='*70}")
for key in ('accuracy', 'precision', 'recall', 'f1', 'top3', 'top5'):
    print(f"  {key:<14} val={val_res[key]*100:>7.2f}%  "
          f"test={test_res[key]*100:>7.2f}%")
print(f"\n  Hierarchical   : 98.40%")
print(f"  This (test)    : {test_res['accuracy']*100:.2f}%")
print(f"  Difference     : {hierarchical_acc - test_res['accuracy']*100:+.2f}%")

# ============================================================================
# SAVE OUTPUTS
# ============================================================================

pd.DataFrame([
    {'split': s, 'accuracy': f"{r['accuracy']*100:.2f}%",
     'f1': f"{r['f1']*100:.2f}%",
     'top3': f"{r['top3']*100:.2f}%",
     'top5': f"{r['top5']*100:.2f}%",
     'vs_hierarchical': f"{hierarchical_acc - r['accuracy']*100:+.2f}%"}
    for s, r in [('val', val_res), ('test', test_res)]
]).to_csv(os.path.join(CHECKPOINT_DIR, 'results.csv'), index=False)

report = classification_report(
    test_res['y_true'], test_res['y_pred'],
    target_names=class_names, zero_division=0
)
with open(os.path.join(CHECKPOINT_DIR, 'classification_report.txt'), 'w') as f:
    f.write("BASELINE 1: ResNet50 Single-Stage TPU - TEST SET\n")
    f.write("Split: 70/15/15\n" + "="*70 + "\n\n")
    f.write(f"Test Accuracy   : {test_res['accuracy']*100:.2f}%\n")
    f.write(f"vs Hierarchical : {hierarchical_acc - test_res['accuracy']*100:+.2f}%\n\n")
    f.write(report)

# Training curves
full_history = load_history(HISTORY_PATH)
if full_history:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ep = range(1, len(full_history['accuracy']) + 1)

    for ax, tr_k, va_k, title, logy in [
        (axes[0,0], 'accuracy',  'val_accuracy',  'Accuracy',    False),
        (axes[0,1], 'loss',      'val_loss',      'Loss',        True),
        (axes[1,0], 'top3_acc',  'val_top3_acc',  'Top-3 Acc',   False),
    ]:
        ax.plot(ep, full_history.get(tr_k, []), label='Train', color='#2e86de')
        ax.plot(ep, full_history.get(va_k, []), label='Val',   color='#ee5a6f')
        ax.axvline(x=WARMUP_EPOCHS, color='red', linestyle='--', alpha=0.6)
        ax.set_title(title); ax.legend(); ax.grid(True, alpha=0.3)
        if logy: ax.set_yscale('log')

    bars = axes[1,1].bar(
        ['Hierarchical', 'ResNet50\nSingle-Stage'],
        [hierarchical_acc, test_res['accuracy']*100],
        color=['#2ecc71', '#3498db']
    )
    axes[1,1].set_ylim([0, 100])
    axes[1,1].set_title('Accuracy Comparison')
    for bar, v in zip(bars, [hierarchical_acc, test_res['accuracy']*100]):
        axes[1,1].text(bar.get_x() + bar.get_width()/2.,
                       bar.get_height() + 1,
                       f'{v:.2f}%', ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(CHECKPOINT_DIR, 'training_curves.png'),
                dpi=300, bbox_inches='tight')

# Confusion matrix
cm = confusion_matrix(test_res['y_true'], test_res['y_pred'])
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'ResNet50 Single-Stage TPU - Test ({test_res["accuracy"]*100:.1f}%)',
          fontsize=14, fontweight='bold')
plt.ylabel('True'); plt.xlabel('Predicted')
plt.xticks(rotation=90, fontsize=8); plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(CHECKPOINT_DIR, 'confusion_matrix.png'),
            dpi=300, bbox_inches='tight')

print(f"\n{'='*70}")
print("ALL DONE")
print(f"{'='*70}")
print(f"  Test Accuracy : {test_res['accuracy']*100:.2f}%")
print(f"  All outputs   : {CHECKPOINT_DIR}")


# ============================================================================
# PERSISTENCE NOTES
# ============================================================================
"""
RESUMING ACROSS KAGGLE SESSIONS (TPU)
=======================================
IMPORTANT: TPU notebooks on Kaggle do NOT persist /kaggle/working.
You must save checkpoints externally after every session.

FILES TO PRESERVE:
  baseline_checkpoints/best_model/      <- SavedModel directory
  baseline_checkpoints/latest_model/    <- SavedModel directory
  baseline_checkpoints/history.json
  baseline_checkpoints/train_state.json
  baseline_checkpoints/finetune_log.csv
  baseline_split/                       <- 70/15/15 split (entire folder)

HOW TO SAVE (run at end of each session):
  !zip -r /kaggle/working/baseline_checkpoints.zip \
       /kaggle/working/baseline_checkpoints/
  !zip -r /kaggle/working/baseline_split.zip \
       /kaggle/working/baseline_split/
  # Then download both from Kaggle Output tab.

HOW TO RESTORE (run at start of next session):
  import zipfile
  with zipfile.ZipFile('/kaggle/input/YOUR-DS/baseline_checkpoints.zip') as z:
      z.extractall('/kaggle/working/')
  with zipfile.ZipFile('/kaggle/input/YOUR-DS/baseline_split.zip') as z:
      z.extractall('/kaggle/working/')
  # Then run this script - auto-detects and resumes.
"""