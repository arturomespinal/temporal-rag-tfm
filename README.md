# Temporal RAG: Predicción y Detección de Anomalías con Recuperación Basada en Índices Vectoriales

**Trabajo de Fin de Máster — Máster en Ciencia de Datos**  
**Autor:** Arturo Miguel Espinal Reyes  
**Director:** Julio Emilio Sandubete Galán  
**Universidad:** Madrid, 2026

---

## Descripción

Este repositorio contiene la implementación completa del sistema **Temporal RAG** (Retrieval-Augmented Generation aplicado a series temporales), propuesto en el TFM como extensión del paradigma RAG textual (Lewis et al., 2020) al dominio secuencial.

La arquitectura integra:
1. Un **encoder Transformer** entrenado con pérdida contrastiva NT-Xent para generar embeddings dinámicos de ventanas temporales
2. Un **índice vectorial FAISS** que actúa como memoria histórica no paramétrica
3. Un **generador híbrido** con cross-attention que fusiona la representación paramétrica con el contexto histórico recuperado

**Hipótesis central validada:** La incorporación de recuperación contextual k-NN mejora el MAE en forecasting en un **9.9%** y el F1 en detección de anomalías en un **+7.6%** respecto al baseline paramétrico puro (condición óptima k=5, benchmarks ETTh1 y MSL).

---

## Estructura del Repositorio

```
temporal-rag-tfm/
│
├── train_contrastive.py   # Fase 3: Entrenamiento del encoder con NT-Xent (§6.3, §7.4)
├── build_index.py         # Fase 4: Construcción y serialización del índice FAISS (§6.4)
├── ablation_study.py      # Evaluación: ablación k ∈ {0,1,5,10} (§7.1–7.3)
├── inference.py           # Fases 5–6: Inferencia batch y streaming (§6.5–6.6)
├── eda.py                 # Cap. 5: EDA completo + figuras PCA, STL, ACF/PACF
│
├── data/                  # Datasets NAB (no incluidos — ver §Datos)
│   └── ambient_temperature_system_failure.csv
│
├── checkpoints/           # Generado por train_contrastive.py
│   └── best_encoder.pt
│
├── index/                 # Generado por build_index.py
│   ├── faiss.index
│   ├── windows.npy
│   └── config.json
│
├── results/               # Generado por ablation_study.py e inference.py
│   └── ablation_results.json
│
├── figures/               # Generado por eda.py
│   ├── fig_pca_embeddings.png    ← Figura 1 de la memoria (§5.3.4)
│   ├── fig_stl.png               ← §5.3.2
│   ├── fig_acf_pacf.png          ← §5.3.3
│   └── fig_distribucion.png      ← §5.3.1
│
└── requirements.txt
```

---

## Pipeline Completo — Orden de Ejecución

### Paso 0: Instalación de dependencias

```bash
git clone https://github.com/arturomespinal/temporal-rag-tfm.git
cd temporal-rag-tfm
pip install -r requirements.txt
```

### Paso 1: EDA — Análisis Exploratorio (Capítulo 5)

```bash
python eda.py \
    --data_path data/ambient_temperature_system_failure.csv \
    --labels_path data/labels.json \
    --output_dir figures/ \
    --window_size 96
```

Genera las figuras del Capítulo 5 en `figures/`. La figura `fig_pca_embeddings.png` corresponde a la **Figura 1** de la memoria (§5.3.4).

### Paso 2: Entrenamiento del Encoder (Capítulo 6 — Fase 3)

```bash
python train_contrastive.py \
    --data_path data/ambient_temperature_system_failure.csv \
    --epochs 50 \
    --batch_size 256 \
    --window_size 96 \
    --d_model 128 \
    --embed_dim 64 \
    --tau 0.2 \
    --lr 3e-4 \
    --checkpoint_dir checkpoints/
```

**Hiperparámetros del TFM (Tabla 7.4.4):**

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| `d_model` | 128 | Balance capacidad/eficiencia |
| `embed_dim` | 64 | Espacio compacto para FAISS |
| `nhead` | 8 | Multi-head estándar (Vaswani et al., 2017) |
| `num_layers` | 3 | Sin sobreajuste en ventanas L=96 |
| `batch_size` | 256 | Maximiza diversidad de negativos NT-Xent |
| `tau` | 0.2 | Gradientes focalizados sin colapso |
| `lr` | 3×10⁻⁴ | Rango efectivo para Transformers |

### Paso 3: Construcción del Índice FAISS (Capítulo 6 — Fase 4)

```bash
python build_index.py \
    --data_path data/ambient_temperature_system_failure.csv \
    --encoder_path checkpoints/best_encoder.pt \
    --index_dir index/ \
    --index_type ivf
```

El índice solo se construye con datos de **entrenamiento** (70% temporal). Ningún dato de validación o test contamina el índice (protocolo §6.9).

### Paso 4: Estudio de Ablación (Capítulo 7)

```bash
python ablation_study.py \
    --data_path data/ambient_temperature_system_failure.csv \
    --labels_path data/labels.json \
    --encoder_path checkpoints/best_encoder.pt \
    --k_values 0 1 5 10 \
    --forecast_horizon 96 \
    --alpha 0.5 \
    --output_json results/ablation_results.json
```

Reproduce las **Tablas 7.1, 7.2 y 7.3** de la memoria. Salida esperada:

```
────────────────────────────────────────────────────────────────────────────────
TABLA 7.1 — Ablación sobre Forecasting (ETTh1, H=96)
────────────────────────────────────────────────────────────────────────────────
   k |      MAE |      MSE |     MASE | Δ MAE vs k=0
   0 |   0.4120 |   0.2980 |   1.0340 |            —
   1 |   0.3890 |   0.2710 |   0.9780 |       -5.6%
   5 |   0.3710 |   0.2490 |   0.9320 |       -9.9%   ← óptimo
  10 |   0.3810 |   0.2640 |   0.9570 |       -7.5%
```

### Paso 5: Inferencia

```bash
# Modo batch — procesa el conjunto de test completo
python inference.py \
    --mode batch \
    --data_path data/ambient_temperature_system_failure.csv \
    --encoder_path checkpoints/best_encoder.pt \
    --index_dir index/ \
    --k 5 \
    --output_path results/inference_output.csv

# Modo streaming — simulación near-real-time con alertas
python inference.py \
    --mode streaming \
    --data_path data/ambient_temperature_system_failure.csv \
    --encoder_path checkpoints/best_encoder.pt \
    --index_dir index/ \
    --k 5 \
    --threshold 0.5 \
    --n_streaming_windows 300
```

---

## Arquitectura del Sistema

```
Ventana de consulta W_q (L=96 puntos)
         │
         ▼ Z-score local (§6.1)
         │
         ▼ TransformerBackbone (§6.3)
         │   ├── PositionalEncoding
         │   ├── TransformerEncoder (3 capas, 8 heads)
         │   └── Global Average Pooling
         │
         z_q ∈ ℝ^128  (L2-normalizado)
         │
         ├──────────────────────────────────────────┐
         │                                          │
         ▼ FAISS IndexIVFFlat (§6.4)               │
         │   └── k=5 vecinos más similares          │
         │                                          │
         ▼ Reconstrucción de ventanas (§6.5)        │
         │                                          │
         ▼ Backbone(ventanas_recuperadas)            │
         │                                          │
         C ∈ ℝ^{k×128}  (contexto)                 │
         │                                          │
         └────────────► CrossAttentionFusion ◄──────┘
                              │
                        z_fused ∈ ℝ^{256}
                         [z_q ‖ c_attn]
                              │
               ┌──────────────┴──────────────┐
               │                             │
               ▼                             ▼
         ForecastHead                  AnomalyHead
         ŷ ∈ ℝ^H                       s_head ∈ [0,1]
                                             │
                              s_isolation (FAISS k-NN)
                                             │
                              s_ensemble = α·s_head + (1-α)·s_iso
```

---

## Resultados Principales

### Forecasting (ETTh1, horizonte H=96)

| Modelo | k | MAE ↓ | MSE ↓ | MASE ↓ |
|--------|---|-------|-------|--------|
| Temporal RAG | 0 (Baseline) | 0.412 | 0.298 | 1.034 |
| Temporal RAG | 1 | 0.389 | 0.271 | 0.978 |
| **Temporal RAG** | **5 (óptimo)** | **0.371** | **0.249** | **0.932** |
| Temporal RAG | 10 | 0.381 | 0.264 | 0.957 |
| iTransformer (SOTA) | — | 0.364 | 0.241 | 0.913 |

### Detección de Anomalías (MSL dataset, Point-Adjust)

| Modelo | k | Precision ↑ | Recall ↑ | F1 ↑ | AUROC ↑ |
|--------|---|-------------|----------|------|---------|
| Temporal RAG | 0 (Baseline) | 0.743 | 0.698 | 0.720 | 0.841 |
| **Temporal RAG** | **5 (óptimo)** | **0.789** | **0.761** | **0.775** | **0.879** |
| FITS (SOTA) | — | 0.801 | 0.778 | 0.789 | 0.893 |

### Latencia de Inferencia

| k | T. Retrieval (ms) | T. Total (ms) | Overhead |
|---|------------------|---------------|---------|
| 0 | 0.0 | 12.3 | — |
| 5 | 3.8 | 16.1 | +30.9% |
| 10 | 7.1 | 19.4 | +57.7% |

---

## Datos

Los datasets utilizados son públicos y deben descargarse por separado:

- **NAB** (ambient_temperature_system_failure): [github.com/numenta/NAB](https://github.com/numenta/NAB)
- **ETTh1/ETTh2**: [github.com/zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)
- **MSL/SMAP**: [NASA Telemetry Anomaly Detection](https://github.com/khundman/telemanom)
- **SMD**: [OmniAnomaly](https://github.com/NetManAIOps/OmniAnomaly)

Colocar los archivos CSV en el directorio `data/`.

---

## Referencias Principales

```bibtex
@inproceedings{lewis2020retrieval,
  title={Retrieval-augmented generation for knowledge-intensive NLP tasks},
  author={Lewis, Patrick and others},
  booktitle={NeurIPS},
  year={2020}
}

@inproceedings{yue2022ts2vec,
  title={{TS2Vec}: Towards universal representation of time series},
  author={Yue, Zhihan and others},
  booktitle={AAAI},
  year={2022}
}

@article{johnson2019billion,
  title={Billion-scale similarity search with {GPUs}},
  author={Johnson, Jeff and Douze, Matthijs and J{\'e}gou, Herv{\'e}},
  journal={IEEE Transactions on Big Data},
  year={2019}
}

@inproceedings{chen2020simple,
  title={A simple framework for contrastive learning of visual representations},
  author={Chen, Ting and others},
  booktitle={ICML},
  year={2020}
}

@inproceedings{xu2022anomaly,
  title={Anomaly {Transformer}: Time series anomaly detection with association discrepancy},
  author={Xu, Jiehui and others},
  booktitle={ICLR},
  year={2022}
}
```

---

## Licencia

MIT License. Ver `LICENSE` para detalles.

---

*Repositorio generado como parte del TFM "Temporal RAG: Predicción y Detección de Anomalías con Recuperación Basada en Índices Vectoriales de Embeddings de Series Temporales", Máster en Ciencia de Datos, Madrid 2026.*
