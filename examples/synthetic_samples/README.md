# 🧪 Exemplos de Dados Sintéticos Gerados (10 Amostras)

Este diretório contém 10 amostras sintéticas geradas a partir do script [`generate_synthetic_dataset.py`](../../generate_synthetic_dataset.py) na branch `synthetic_dataset`.

O gerador sintetiza pares **Figura Científica (JPEG) + Tabela Canônica em Markdown**, seguindo as especificações da Task 2 do **Sci-ImageMiner** no domínio de deposição e corrosão em escala atômica (*Atomic Layer Deposition - ALD* e *Atomic Layer Etching - ALE*).

---

## 📊 Estatísticas das 10 Amostras

* **Total de Imagens:** 10 arquivos JPEG em [`images/`](images/)
* **Distribuição de Dificuldade:**
  * **Easy (5):** `ald_growth`, `ale_mass_change`, `publication_bar`, `single_saturation`, `surface_coverage`
  * **Medium (4):** `barrier_methods`, `electrical_dual_axis`, `emission_spectra`, `temperature_series`
  * **Hard (1):** `reaction_pathways`
* **Divisão dos Splits:**
  * **Treino (`train.jsonl`):** 6 amostras
  * **Validação (`val.jsonl`):** 1 amostra
  * **Teste (`test.jsonl`):** 3 amostras

---

## 🔍 Catálogo das Amostras Geradas

### 1. `synthetic__atomic_layer_deposition__experimental_usecase__10__fig_1` (Easy - ALD Growth)
* **Split:** Teste
* **Arquivo:** [`images/synthetic__atomic_layer_deposition__experimental_usecase__10__fig_1.jpg`](images/synthetic__atomic_layer_deposition__experimental_usecase__10__fig_1.jpg)
* **Tabela Ground Truth:**
| Number of ALD Cycles | Film Thickness (nm) |
|---|---|
| 0 | 0 |
| 10 | 1.08 |
| 20 | 1.76 |
| 30 | 2.51 |
| 40 | 3.3 |
| 50 | 4.1 |
| 60 | 4.88 |
| 70 | 5.67 |

---

### 2. `synthetic__atomic_layer_etching__experimental_usecase__3__figure_1` (Easy - ALE Mass Change)
* **Split:** Teste
* **Arquivo:** [`images/synthetic__atomic_layer_etching__experimental_usecase__3__figure_1.jpg`](images/synthetic__atomic_layer_etching__experimental_usecase__3__figure_1.jpg)
* **Tabela Ground Truth:**
| Temperature (°C) | Mass Change Per Cycle (ng/cm²) |
|---|---|
| 100 | -2.3 |
| 150 | -42.5 |
| 200 | -90.2 |
| 250 | -138.6 |
| 300 | -180.4 |

---

### 3. `synthetic__atomic_layer_deposition__experimental_usecase__2__fig_1` (Easy - Surface Coverage)
* **Split:** Teste
* **Arquivo:** [`images/synthetic__atomic_layer_deposition__experimental_usecase__2__fig_1.jpg`](images/synthetic__atomic_layer_deposition__experimental_usecase__2__fig_1.jpg)
* **Tabela Ground Truth:**
| Number of Cycles | Ti Atom Surface Coverage (×10¹⁵/cm²) |
|---|---|
| 0 | 0.0 |
| 10 | 3.8 |
| 20 | 8.4 |
| 30 | 14.2 |
| 40 | 19.1 |
| 50 | 22.8 |

---

### 4. `synthetic__atomic_layer_deposition__experimental_usecase__8__fig_1` (Easy - Single Saturation)
* **Split:** Treino
* **Arquivo:** [`images/synthetic__atomic_layer_deposition__experimental_usecase__8__fig_1.jpg`](images/synthetic__atomic_layer_deposition__experimental_usecase__8__fig_1.jpg)
* **Tabela Ground Truth:**
| Exposure Time (s) | Mass Gain per Cycle (ng/cm²) |
|---|---|
| 0.0 | 0.36 |
| 0.2 | 9.32 |
| 0.5 | 17.86 |
| 1.0 | 23.95 |
| 2.0 | 25.17 |
| 4.0 | 24.89 |

---

### 5. `synthetic__atomic_layer_deposition__experimental_usecase__9__figure_1` (Easy - Publication Bar)
* **Split:** Treino
* **Arquivo:** [`images/synthetic__atomic_layer_deposition__experimental_usecase__9__figure_1.jpg`](images/synthetic__atomic_layer_deposition__experimental_usecase__9__figure_1.jpg)
* **Tabela Ground Truth:**
| Year | Publications per Year |
|---|---|
| 1992 | 0 |
| 1994 | 1 |
| 1996 | 1 |
| 1998 | 2 |
| 2000 | 4 |
| 2002 | 8 |
| 2004 | 15 |
| 2006 | 29 |
| 2008 | 45 |
| 2010 | 70 |

---

### 6. `synthetic__atomic_layer_deposition__experimental_usecase__5__figure_1` (Medium - Barrier Methods)
* **Split:** Validação
* **Arquivo:** [`images/synthetic__atomic_layer_deposition__experimental_usecase__5__figure_1.jpg`](images/synthetic__atomic_layer_deposition__experimental_usecase__5__figure_1.jpg)
* **Tabela Ground Truth:**
| Film Thickness (nm) | PE-ALD TaN | PVD TaN | PE-ALD Ta | PVD Ta |
|---|---|---|---|---|
| 0 | 568 | 593 | 590 | 583 |
| 5 | 657 | 681 | 682 | 666 |
| 10 | 732 | 779 | 762 | 762 |
| 15 | 808 | 851 | 823 | 829 |
| 20 | 857 | 913 | 884 | 887 |

---

### 7. `synthetic__atomic_layer_deposition__experimental_usecase__1__figure_1` (Medium - Emission Spectra)
* **Split:** Treino
* **Arquivo:** [`images/synthetic__atomic_layer_deposition__experimental_usecase__1__figure_1.jpg`](images/synthetic__atomic_layer_deposition__experimental_usecase__1__figure_1.jpg)
* **Tabela Ground Truth:**
| Wavelength (nm) | O₂ Intensity (a.u.) | H₂ Intensity (a.u.) | N₂ Intensity (a.u.) |
|---|---|---|---|
| 100 | 0.043 | 0.064 | 0.039 |
| 150 | 0.043 | 0.060 | 0.054 |
| 200 | 0.133 | 0.060 | 0.054 |
| 250 | 0.046 | 0.058 | 0.050 |
| 300 | 0.057 | 0.165 | 0.057 |
| 350 | 0.060 | 0.061 | 0.063 |
| 400 | 0.061 | 0.055 | 0.061 |
| 450 | 0.055 | 0.060 | 0.059 |
| 500 | 0.056 | 0.062 | 0.064 |

---

### 8. `synthetic__atomic_layer_deposition__experimental_usecase__6__fig_1` (Medium - Electrical Dual Axis)
* **Split:** Treino
* **Arquivo:** [`images/synthetic__atomic_layer_deposition__experimental_usecase__6__fig_1.jpg`](images/synthetic__atomic_layer_deposition__experimental_usecase__6__fig_1.jpg)
* **Tabela Ground Truth:**
| Growth Temperature (°C) | Dielectric Constant, k | Breakdown Field (MV/cm) |
|---|---|---|
| 33 | 8.05 | 3.64 |
| 58 | 7.27 | 3.83 |
| 80 | 7.28 | 3.96 |
| 102 | 7.91 | 4.19 |
| 125 | 8.21 | 4.31 |
| 177 | 7.57 | 4.41 |

---

### 9. `synthetic__atomic_layer_etching__experimental_usecase__4__fig_1` (Medium - Temperature Series)
* **Split:** Treino
* **Arquivo:** [`images/synthetic__atomic_layer_etching__experimental_usecase__4__fig_1.jpg`](images/synthetic__atomic_layer_etching__experimental_usecase__4__fig_1.jpg)
* **Tabela Ground Truth:**
| Number of Cycles | 128°C | 160°C | 196°C |
|---|---|---|---|
| 0 | 136.5 | 144.0 | 138.0 |
| 2 | 134.8 | 141.1 | 135.5 |
| 4 | 133.7 | 137.3 | 133.6 |
| 6 | 132.0 | 133.9 | 130.9 |
| 8 | 130.4 | 131.2 | 129.2 |
| 10 | 128.8 | 127.8 | 127.3 |

---

### 10. `synthetic__atomic_layer_deposition__simulation_usecase__7__figure_1` (Hard - Reaction Energy Pathways)
* **Split:** Treino
* **Arquivo:** [`images/synthetic__atomic_layer_deposition__simulation_usecase__7__figure_1.jpg`](images/synthetic__atomic_layer_deposition__simulation_usecase__7__figure_1.jpg)
* **Tabela Ground Truth:**
| Reaction Step | Path A | Path B | Path C |
|---|---|---|---|
| Reactant | 0.0 | 0.0 | 0.0 |
| Complex | -12.9 | -13.2 | -12.9 |
| TS | 44.1 | 47.3 | 51.2 |
| Intermediate | 12.0 | 14.6 | 17.8 |
| Product | -28.4 | -26.1 | -24.2 |
