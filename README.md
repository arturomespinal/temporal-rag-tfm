# Temporal RAG: Prediction and Anomaly Detection with Vector Index Retrieval

> TFM — Máster Universitario en Ciencia de Datos  
> Autor: Arturo Miguel Espinal Reyes | Director: Julio Emilio Sandubete Galán

---

## Estructura del Repositorio

```
temporal_rag/
├── encoder/
│   └── train_contrastive.py     # Entrenamiento del encoder (NT-Xent)
├── experiments/
│   └── ablation_study.py        # Experimento de ablación k=0,1,5,10
├── indexing/                    # (Cap. 5) Construcción del índice FAISS
├── retrieval/                   # (Cap. 5) Pipeline de recuperación
├── forecasting/                 # (Cap. 6) Módulo de predicción
├── anomaly/                     # (Cap. 6) Módulo de detección
└── results/                     # Tablas CSV y LaTeX generadas
```

## Instalación

```bash
pip install torch faiss-cpu numpy pandas scikit-learn
```

## Uso Rápido

```bash
# 1. Entrenar el encoder contrastivo
python encoder/train_contrastive.py \
    --data-path data/ETTh1_train.npy \
    --epochs 50 --window-size 96 --embed-dim 64

# 2. Ejecutar experimento de ablación
python experiments/ablation_study.py \
    --encoder-path checkpoints/encoder_best.pt \
    --index-path checkpoints/faiss.index \
    --test-data data/ETTh1_test.npy \
    --task forecasting \
    --k-values 0 1 5 10
```

## Referencias Clave

- Chen et al. (2020). SimCLR. ICML 2020.
- Yue et al. (2022). TS2Vec. AAAI 2022.
- Lewis et al. (2020). RAG. NeurIPS 2020.
- Johnson et al. (2021). FAISS. IEEE TPAMI.
