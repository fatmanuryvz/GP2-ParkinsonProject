"""
NeuroScan — Parkinson Erken Tanı Sistemi
MobileNetV2 Transfer Learning
Yeni Dataset: 3264 görüntü (1632 Healthy + 1632 Parkinson)
"""
import os, json, shutil, warnings
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve,
                              accuracy_score, precision_score,
                              recall_score, f1_score)
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

tf.random.set_seed(42)
np.random.seed(42)

IMG_SIZE   = (224, 224)
BATCH      = 32
MODEL_PATH = "parkinson_cnn_model.h5"
DATA_DIR   = "data_new/Dataset/Dataset"   # Healthy / Parkinson klasörleri burada

print("="*60)
print("  NeuroScan — MobileNetV2 (3264 Görüntü)")
print("="*60)

# ── Train/Val/Test olarak ayır ──
def split_dataset(data_dir, output_dir="data_split", test_size=0.15, val_size=0.15):
    """Veriyi train/val/test olarak böl ve kopyala"""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    classes = ['Healthy', 'Parkinson']
    all_counts = {}

    for cls in classes:
        src = os.path.join(data_dir, cls)
        files = [f for f in os.listdir(src) if f.lower().endswith(('.png','.jpg','.jpeg'))]
        
        # Train/val/test böl
        train_val, test = train_test_split(files, test_size=test_size, random_state=42)
        train, val      = train_test_split(train_val, test_size=val_size/(1-test_size), random_state=42)

        all_counts[cls] = {'train': len(train), 'val': len(val), 'test': len(test)}

        for split_name, split_files in [('training', train), ('validation', val), ('testing', test)]:
            out = os.path.join(output_dir, split_name, cls.lower())
            os.makedirs(out, exist_ok=True)
            for fname in split_files:
                shutil.copy2(os.path.join(src, fname), os.path.join(out, fname))

    return output_dir, all_counts

print("\n📁 Veri bölünüyor (Train %70 / Val %15 / Test %15)...")
split_dir, counts = split_dataset(DATA_DIR)

print("\n  Veri dağılımı:")
for cls, c in counts.items():
    print(f"  {cls}: Train={c['train']} | Val={c['val']} | Test={c['test']}")

TRAIN_DIR = os.path.join(split_dir, 'training')
VAL_DIR   = os.path.join(split_dir, 'validation')
TEST_DIR  = os.path.join(split_dir, 'testing')

# ── VERİ GENERATORLERİ ──
train_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.10,
    height_shift_range=0.10,
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.15,
    shear_range=0.10,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
val_aug  = ImageDataGenerator(rescale=1./255)
test_aug = ImageDataGenerator(rescale=1./255)

train_gen = train_aug.flow_from_directory(
    TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='binary', seed=42, shuffle=True)

val_gen = val_aug.flow_from_directory(
    VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH,
    class_mode='binary', shuffle=False)

test_gen = test_aug.flow_from_directory(
    TEST_DIR, target_size=IMG_SIZE, batch_size=1,
    class_mode='binary', shuffle=False)

class_names = list(train_gen.class_indices.keys())
print(f"\n  Sınıflar: {train_gen.class_indices}")
print(f"  Train: {train_gen.samples} | Val: {val_gen.samples} | Test: {test_gen.samples}")

# ── MODEL ──
base = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE+(3,), include_top=False, weights='imagenet')
base.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE+(3,))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(64, activation='relu')(x)
x = layers.Dropout(0.2)(x)
output = layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, output, name="ParkinsonMobileNet_v3")

# ── AŞAMA 1: Head eğitimi ──
print("\n[AŞAMA 1] Head katmanları eğitiliyor...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy', metrics=['accuracy'])

cb1 = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=8,
                             restore_best_weights=True, mode='max', verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                 patience=4, min_lr=1e-6, verbose=1),
]
h1 = model.fit(train_gen, epochs=20, validation_data=val_gen,
                callbacks=cb1, verbose=1)
print(f"  En iyi val accuracy: {max(h1.history['val_accuracy'])*100:.2f}%")

# ── AŞAMA 2: Fine-tuning ──
print("\n[AŞAMA 2] Fine-tuning (son 40 katman)...")
base.trainable = True
for layer in base.layers[:-40]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-6),
    loss='binary_crossentropy', metrics=['accuracy'])

cb2 = [
    callbacks.EarlyStopping(monitor='val_accuracy', patience=15,
                             restore_best_weights=True, mode='max', verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,
                                 patience=6, min_lr=1e-8, verbose=1),
    callbacks.ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                               save_best_only=True, mode='max', verbose=1),
]
h2 = model.fit(train_gen, epochs=50, validation_data=val_gen,
                callbacks=cb2, verbose=1)
best_val = max(h2.history['val_accuracy'])*100
print(f"  En iyi val accuracy: {best_val:.2f}%")

# ── TEST ──
print("\n[TEST DEĞERLENDİRMESİ]")
test_gen.reset()
preds    = model.predict(test_gen, verbose=0).flatten()
y_pred   = (preds > 0.5).astype(int)
y_true   = test_gen.classes
y_scores = preds

print(classification_report(y_true, y_pred, target_names=class_names))

cm   = confusion_matrix(y_true, y_pred)
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred, zero_division=0)
f1   = f1_score(y_true, y_pred, zero_division=0)
spec = cm[0,0]/cm[0].sum() if cm[0].sum()>0 else 0
try:    roc_v = roc_auc_score(y_true, y_scores)
except: roc_v = 0.0

print(f"\n  Accuracy:    {acc*100:.2f}%")
print(f"  Precision:   {prec*100:.2f}%")
print(f"  Recall:      {rec*100:.2f}%")
print(f"  F1-Score:    {f1*100:.2f}%")
print(f"  Specificity: {spec*100:.2f}%")
print(f"  AUC-ROC:     {roc_v:.4f}")

# ── GRAFİKLER ──
all_acc  = h1.history['accuracy']     + h2.history['accuracy']
all_val  = h1.history['val_accuracy'] + h2.history['val_accuracy']
all_loss = h1.history['loss']         + h2.history['loss']
all_vl   = h1.history['val_loss']     + h2.history['val_loss']
ep = range(1, len(all_acc)+1)
split_ep = len(h1.history['accuracy'])

fig, axes = plt.subplots(1,2,figsize=(14,5))
fig.patch.set_facecolor('#0d1117')
for ax in axes:
    ax.set_facecolor('#161b22')
    for sp in ax.spines.values(): sp.set_color('#30363d')
    ax.tick_params(colors='#8b949e')

axes[0].plot(ep,[v*100 for v in all_acc],color='#4ade80',lw=2,label='Train')
axes[0].plot(ep,[v*100 for v in all_val],color='#63b3ed',lw=2,ls='--',label='Val')
axes[0].axvline(x=split_ep,color='#f0883e',ls=':',lw=1.5,label='Fine-tune')
axes[0].set_title('Doğruluk',color='#e6edf3',fontweight='bold')
axes[0].set_xlabel('Epoch',color='#8b949e'); axes[0].set_ylabel('Accuracy (%)',color='#8b949e')
axes[0].legend(facecolor='#21262d',labelcolor='#e6edf3'); axes[0].grid(alpha=0.15)

axes[1].plot(ep,all_loss,color='#f0883e',lw=2,label='Train')
axes[1].plot(ep,all_vl,color='#d2a8ff',lw=2,ls='--',label='Val')
axes[1].axvline(x=split_ep,color='#63b3ed',ls=':',lw=1.5)
axes[1].set_title('Kayıp',color='#e6edf3',fontweight='bold')
axes[1].set_xlabel('Epoch',color='#8b949e'); axes[1].set_ylabel('Loss',color='#8b949e')
axes[1].legend(facecolor='#21262d',labelcolor='#e6edf3'); axes[1].grid(alpha=0.15)
plt.tight_layout()
plt.savefig('training_curves.png',dpi=150,bbox_inches='tight',facecolor='#0d1117')
plt.close(); print("  ✓ training_curves.png")

fig,ax = plt.subplots(figsize=(6,5))
fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#0d1117')
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',
            xticklabels=class_names,yticklabels=class_names,
            ax=ax,annot_kws={'size':16,'weight':'bold'},
            linewidths=2,linecolor='#0d1117')
ax.set_xlabel('Tahmin',fontsize=12,color='#e6edf3')
ax.set_ylabel('Gerçek',fontsize=12,color='#e6edf3')
ax.set_title('Confusion Matrix',color='#e6edf3',fontweight='bold')
ax.tick_params(colors='#8b949e')
plt.tight_layout()
plt.savefig('confusion_matrix.png',dpi=150,bbox_inches='tight',facecolor='#0d1117')
plt.close(); print("  ✓ confusion_matrix.png")

fpr,tpr,_ = roc_curve(y_true,y_scores)
fig,ax = plt.subplots(figsize=(6,5))
fig.patch.set_facecolor('#0d1117'); ax.set_facecolor('#161b22')
ax.plot(fpr,tpr,color='#63b3ed',lw=2.5,label=f'AUC = {roc_v:.4f}')
ax.fill_between(fpr,tpr,alpha=0.15,color='#63b3ed')
ax.plot([0,1],[0,1],color='#f85149',ls='--',lw=1.5,label='Random')
ax.set_xlabel('False Positive Rate',color='#8b949e')
ax.set_ylabel('True Positive Rate',color='#8b949e')
ax.set_title('ROC Eğrisi',color='#e6edf3',fontweight='bold')
ax.legend(facecolor='#21262d',labelcolor='#e6edf3'); ax.grid(alpha=0.15)
for sp in ax.spines.values(): sp.set_color('#30363d')
ax.tick_params(colors='#8b949e')
plt.tight_layout()
plt.savefig('roc_curve.png',dpi=150,bbox_inches='tight',facecolor='#0d1117')
plt.close(); print("  ✓ roc_curve.png")

results = {
    "accuracy": round(acc,4), "precision": round(prec,4),
    "recall": round(rec,4), "f1_score": round(f1,4),
    "specificity": round(spec,4), "auc_roc": round(roc_v,4),
    "best_val_accuracy": round(best_val,2),
    "class_names": class_names,
    "class_indices": train_gen.class_indices,
    "dataset": "3264 görüntü (1632 Healthy + 1632 Parkinson)"
}
with open('model_results.json','w') as f:
    json.dump(results, f, indent=2)
print("  ✓ model_results.json")

print("\n"+"="*60)
print(f"  ✅ EĞİTİM TAMAMLANDI!")
print(f"  🏆 Test Doğruluğu: {acc*100:.2f}%")
print(f"  📈 AUC-ROC: {roc_v:.4f}")
print("="*60)