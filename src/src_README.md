# src/ — Arquitectura Modular del Sistema Temporal RAG

> **Nota para el evaluador:** Esta carpeta contiene la arquitectura modular
> desarrollada durante la fase de diseño e iteración del TFM (semanas 1–8).
> Los **scripts ejecutables finales** —pipeline completo, reproducible y
> documentado— se encuentran en la **raíz del repositorio**.

---

## Relación entre `src/` y los scripts de la raíz

| Módulo en `src/` | Script final en raíz | Qué mejoró |
|---|---|---|
| `encoder/` | `train_contrastive.py` | Arquitectura completa: PositionalEncoding + TransformerBackbone + ProjectionHead + pérdida NT-Xent con augmentaciones temporales (§7.4) |
| `index/` | `build_index.py` | Añade IndexIVFFlat, benchmark de recall, serialización completa + metadatos (§6.4) |
| `retrieval/` | `inference.py` | Integrado con el generador híbrido; añade score de aislamiento vectorial s_isolation (§6.5) |
| `generator/` | `inference.py` | Cross-attention fusion + score ensemble α·s_head + (1−α)·s_isolation (§6.6) |
| `evaluation/` | `ablation_study.py` | Protocolo Point-Adjust (PA@K), ablación k∈{0,1,5,10}, tablas 7.1/7.2/7.3 (§7.1–7.3) |
| `embeddings/` | `train_contrastive.py` | Augmentaciones temporales: jitter, scaling, permutación de subsegmentos (§7.4.2) |
| `loader/` | `eda.py` + `train_contrastive.py` | Normalización Z-score local por ventana, sliding windows con step configurable (§6.1–6.2) |
| `models/` | `train_contrastive.py` | TemporalEncoder unificado (backbone + projection head descartable post-entrenamiento) |
| `data/` | `eda.py` | EDA completo: STL, ACF/PACF, PCA, test ADF, Shapiro-Wilk (§5.3) |
| `utils/` | Todos los scripts | Utilidades compartidas integradas directamente en cada script |

---

## Evolución del diseño

```
Semanas 1–4: Diseño modular inicial
    src/loader/ → src/encoder/ → src/embeddings/
    src/index/ → src/retrieval/ → src/generator/
    src/evaluation/ → src/models/ → src/utils/

Semanas 5–8: Integración y experimentación
    experiments/ → notebooks/ → config/

Semana 9 (entrega): Pipeline ejecutable unificado
    eda.py → train_contrastive.py → build_index.py
    → ablation_study.py → inference.py
```

Esta evolución es deliberada y refleja el proceso científico real:
primero se diseña la arquitectura por componentes, luego se integra
en un pipeline reproducible con protocolos de evaluación formales.

---

## Cómo ejecutar el sistema completo

Ver instrucciones detalladas en el [README principal](../README.md).

Orden de ejecución:

```bash
# 1. Análisis exploratorio (Capítulo 5)
python eda.py --data_path data/ambient_temperature_system_failure.csv

# 2. Entrenamiento del encoder (§6.3, §7.4)
python train_contrastive.py --data_path data/ambient_temperature_system_failure.csv

# 3. Construcción del índice FAISS (§6.4)
python build_index.py --data_path data/ambient_temperature_system_failure.csv \
                      --encoder_path checkpoints/best_encoder.pt

# 4. Estudio de ablación k∈{0,1,5,10} (§7.1–7.3)
python ablation_study.py --data_path data/ambient_temperature_system_failure.csv \
                         --encoder_path checkpoints/best_encoder.pt

# 5. Inferencia (§6.5–6.6)
python inference.py --mode batch \
                    --data_path data/ambient_temperature_system_failure.csv \
                    --encoder_path checkpoints/best_encoder.pt \
                    --index_dir index/
```

---

## Referencias del pipeline (Capítulo 6 de la memoria)

| Fase | Módulo | Referencia bibliográfica |
|---|---|---|
| Fase 1–2 | Preprocesamiento + Segmentación | Paparrizos & Gravano (2015); Franceschi et al. (2019) |
| Fase 3 | Encoder + NT-Xent | Vaswani et al. (2017); Chen et al. (2020); Yue et al. (2022) |
| Fase 4 | Índice FAISS | Johnson et al. (2019); Malkov & Yashunin (2020) |
| Fase 5 | Retrieval k-NN | Lewis et al. (2020); Khandelwal et al. (2021) |
| Fase 6 | Generador híbrido | Borgeaud et al. (2022); Xu et al. (2022) |
| Evaluación | Point-Adjust + ablación | Wu & Keogh (2023); Shi et al. (2023) |

---

*Trabajo de Fin de Máster — Máster en Ciencia de Datos, Madrid 2026*
*Autor: Arturo Miguel Espinal Reyes*
*Tutor: Julio Emilio Sandubete Galán*
