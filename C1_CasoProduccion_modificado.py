# %% [markdown]
# # Predicción de defectos en producción - XYZ Inc
# Esp. Inteligencia Artificial - CADI Introducción a Machine Learning (2026-2)
# Autores: Jahir Avila y Laura Lopez
#
# NOTA: se trabaja con datos sintéticos (simulados) con semilla fija, porque el script
# original C1_CasoProduccion.py no estaba disponible. Los resultados son reproducibles.

# %% Librerías
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve)

# %% Generación de datos sintéticos (2000 lotes)
rng = np.random.default_rng(42)
n = 2000
df = pd.DataFrame({
    "temperatura": rng.normal(75, 8, n),
    "presion": rng.normal(100, 12, n),
    "velocidad": rng.normal(50, 6, n),
    "horas_operacion": rng.uniform(0, 16, n),
    "vibracion": rng.gamma(4, 0.5, n),
})
z = (-4.6 + 0.07*(df.temperatura-75) + 0.03*(df.presion-100) + 0.05*(df.velocidad-50)
     + 0.16*df.horas_operacion + 0.55*df.vibracion + 0.0006*(df.temperatura-75)**2*(df.vibracion))
p = 1/(1+np.exp(-z))
df["defectuoso"] = (rng.uniform(size=n) < p).astype(int)
print(df.describe().round(2))
print(df.defectuoso.value_counts(), df.defectuoso.mean())

# %% Partición estratificada 80/20
X, y = df.drop(columns="defectuoso"), df["defectuoso"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %% Modelos
modelos = {
    "Regresión Logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42)),
    "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7)),
    "Naive Bayes": GaussianNB(),
}

# %% Entrenamiento y métricas
res, cms, rocs = {}, {}, {}
for nom, m in modelos.items():
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    prob = m.predict_proba(Xte)[:, 1]
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    res[nom] = dict(accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred, zero_division=0),
                    recall=recall_score(yte, pred), f1=f1_score(yte, pred),
                    especificidad=tn/(tn+fp), roc_auc=roc_auc_score(yte, prob),
                    TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))
    cms[nom] = [[int(tn), int(fp)], [int(fn), int(tp)]]
    rocs[nom] = roc_curve(yte, prob)
tabla = pd.DataFrame(res).T
print(tabla.round(3))

# %% Gráficas
fig, axs = plt.subplots(2, 3, figsize=(11, 6.5))
for ax, (nom, cm) in zip(axs.ravel(), cms.items()):
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i][j] > 150 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No def.", "Def."]); ax.set_yticklabels(["No def.", "Def."])
    ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title(nom, fontsize=10)
plt.tight_layout(); plt.savefig("fig_confusion.png", dpi=160); plt.close()

met = ["accuracy", "precision", "recall", "f1"]
fig, ax = plt.subplots(figsize=(10, 4.5))
w = 0.2
for k, mt in enumerate(met):
    ax.bar(np.arange(len(res)) + k*w, [res[m][mt] for m in res], w, label=mt)
ax.set_xticks(np.arange(len(res)) + 1.5*w); ax.set_xticklabels(res.keys(), fontsize=8)
ax.set_ylim(0, 1.15); ax.legend(ncol=4, loc="upper center"); ax.set_title("Comparación de métricas por modelo")
plt.tight_layout(); plt.savefig("fig_metricas.png", dpi=160); plt.close()

fig, ax = plt.subplots(figsize=(6, 5))
for nom, (fpr, tpr, _) in rocs.items():
    ax.plot(fpr, tpr, label=f"{nom} (AUC={res[nom]['roc_auc']:.2f})")
ax.plot([0, 1], [0, 1], "k--"); ax.set_xlabel("Tasa de falsos positivos"); ax.set_ylabel("Recall (TPR)")
ax.set_title("Curvas ROC"); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig_roc.png", dpi=160); plt.close()

# %% Experimento B: ajuste del umbral de decisión (0.25) para priorizar recall
UMBRAL = 0.25
resB = {}
for nom, m in modelos.items():
    prob = m.predict_proba(Xte)[:, 1]
    pred = (prob >= UMBRAL).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    resB[nom] = dict(accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred, zero_division=0),
                     recall=recall_score(yte, pred), f1=f1_score(yte, pred), especificidad=tn/(tn+fp),
                     TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))
print(pd.DataFrame(resB).T.round(3))

json.dump(dict(res=res, resB=resB, umbral=UMBRAL, prevalencia=float(y.mean()), n_test=len(yte), pos_test=int(yte.sum())),
          open("resultados.json", "w"), indent=1)
