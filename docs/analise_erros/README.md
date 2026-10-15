# 🔬 Análise de Erros no Dev: Nota Oficial por Painel

> **Benchmark:** [SciKnowOrg/Sci-ImageMiner](https://huggingface.co/datasets/SciKnowOrg/Sci-ImageMiner) (ICDAR 2026, Task 2)
> **Conjunto:** **dev/validação**: 104 figuras, **167 painéis** com tabela. O teste fica fora, para não viciar decisões.
> **Métrica:** a **oficial** da competição (`½·RMS + ½·TEDS`, script `sci-imageminer_evaluation.py`), **painel por painel**. É a mesma régua do leaderboard.
> **Modelos:** `Qwen2.5-VL-3B` + adapter LoRA do HF (`lucasoc/sci-image-models`) e o mesmo 3B **sem treino**, ambos em resolução original. O prompt é o da `main` de 14/10, que pede uma tabela por painel e é diferente do prompt com que o adapter foi treinado.
> **Branch:** `analise-de-erros` · gerado em 15/10/2026

> ⚠️ **As categorias de erro vêm de regras automáticas e podem errar** (o caso 06 mostra uma regra errando). Os números de nota são exatos; os de categoria são uma estimativa.

---

## 📈 Nota oficial

| Modelo | Nota | RMS (números) | TEDS (estrutura) |
| :--- | :---: | :---: | :---: |
| **3B + LoRA** | **35.0** | 15.6 | 54.5 |
| **3B base (sem treino)** | **23.2** | 8.8 | 37.6 |
| 3B + LoRA, só painéis de gráfico (124) | **43.9** | 21.0 | 66.8 |

- A nota bate **exatamente** com o script oficial rodado nos mesmos arquivos.
- A métrica local do repo deu `icdar_score` 0.43 para o mesmo modelo. Ela não é comparável ao leaderboard. Como referência (que foi no teste, não no dev): baseline Qwen3-VL-8B sem treino 35.97, 1º lugar 41.81.
- O fine-tune ganha **+11.8 pontos** sobre o modelo sem treino.

---

## 🧭 Os 4 achados principais

1. **Um único artigo distorce o dev.** O `atomic-layer-deposition/simulation-usecase/13` tem **42 dos 167 painéis** e custa **21.3 pontos**. São figuras de 6 painéis de diagramas cujo gabarito é **texto descritivo**. Dos 28 painéis "esquecidos", 27 são dele ([caso 01](./01_diagrama_sem_dados_6_paineis.md)). **Só nos painéis de gráfico, o LoRA tira 43.9.**
2. **O modelo monta a tabela, mas erra os números.** TEDS 54.5 contra RMS 15.6. Os padrões que mais aparecem são número de linhas diferente do gabarito, colunas/séries divergentes e grade regular inventada no lugar da curva real ([casos 02 a 05](#-tabela-resumo-dos-casos)).
3. **O gabarito tem ruído real, de formato e de valor.** Há células com `~` e texto ([caso 07](./07_gabarito_com_aproximacoes.md)), valores errados ([caso 06](./06_valor_constante_gabarito_errado.md)) e eixo X errado ([caso 08](./08_metrica_generosa.md)). Isso confirma o que o Lucas viu nos casos 3 e 266.
4. **A métrica oficial é generosa em eixos estreitos.** Uma curva errada de band gap tirou 97 ([caso 08](./08_metrica_generosa.md)).

---

## 📋 Tabela resumo dos casos

| # | Caso | Categoria da falha | Nota | RMS | TEDS |
| :-: | :--- | :--- | :---: | :---: | :---: |
| **01** | [Diagrama sem dados: gabarito de texto em 6 painéis](./01_diagrama_sem_dados_6_paineis.md) | `Painéis que não são gráfico (gabarito descritivo)` | `24.0` | `0.0` | `48.0` |
| **02** | [Tendência invertida](./02_tendencia_invertida.md) | `Curva lida de trás para frente` | `27.5` | `0.0` | `55.0` |
| **03** | [Várias séries espremidas numa célula](./03_series_numa_celula.md) | `Estrutura: legenda virou texto dentro da célula` | `21.2` | `0.0` | `42.3` |
| **04** | [Rampa inventada no lugar de patamares (e bbox trocado no metadata)](./04_rampa_inventada_bbox_trocado.md) | `Linearização de curva em degraus` | `39.1` | `4.0` | `74.3` |
| **05** | [Eixo X sem rótulos (perfil de energia de reação)](./05_eixo_categorico_sem_rotulo.md) | `Linhas da tabela vieram dos ticks do eixo Y` | `33.9` | `0.0` | `67.9` |
| **06** | [Valor constante, gabarito errado e regra errada](./06_valor_constante_gabarito_errado.md) | `Três erros no mesmo painel` | `47.2` | `0.0` | `94.4` |
| **07** | [Gabarito com aproximações e texto](./07_gabarito_com_aproximacoes.md) | `Gabarito suspeito` | `23.2` | `0.0` | `46.5` |
| **08** | [Métrica generosa: curva errada com nota 97](./08_metrica_generosa.md) | `Limite da métrica oficial` | `97.3` | `96.9` | `97.8` |

---

## 💸 Onde a nota se perde (custo por categoria)

**Pontos perdidos** = soma de `100 − nota` dos painéis da categoria ÷ 167. É quanto da nota média aquela categoria leva. Um painel pode ter várias categorias, então as somas se sobrepõem. **Só essa categoria** conta só os painéis em que ela é o único erro.

| Categoria | Painéis (LoRA) | Pontos perdidos (LoRA) | Só essa categoria (LoRA) | Painéis (base) | Pontos perdidos (base) |
|:---|:---:|:---:|:---:|:---:|:---:|
| amostragem_linhas | 63 | 24.9 | 3.7 | 47 | 20.4 |
| colunas_divergentes | 56 | 22.5 | 7.1 | 42 | 18.2 |
| grade_regular_inventada | 53 | 19.7 | 2.6 | 26 | 9.6 |
| painel_faltando | 28 | 16.8 | 16.8 | 63 | 37.7 |
| cabecalho_desalinhado | 15 | 6.1 | 0.8 | 15 | 6.3 |
| escala_magnitude | 3 | 1.2 | 0.3 | 0 | 0.0 |
| tabela_invalida | 2 | 1.2 | 1.2 | 7 | 4.2 |
| numero_nao_numerico | 1 | 0.4 | 0.0 | 5 | 1.7 |

- **37 painéis** do LoRA não caíram em nenhuma categoria. Eles tiram 61.6 em média: o erro é só nos valores.
- `painel_faltando` é o maior custo exclusivo, mas, como dito acima, vem quase todo de um artigo.

---

## 🔎 Recortes

### Por número de painéis na figura
| Figura | Painéis | Nota LoRA | RMS LoRA | TEDS LoRA | Nota base |
|:---|:---:|:---:|:---:|:---:|:---:|
| 2+ painéis | 99 | 33.0 | 17.7 | 48.2 | 22.0 |
| 1 painel | 68 | 38.1 | 12.4 | 63.7 | 24.9 |

### Por tamanho do gabarito
| Gabarito | Painéis | Nota LoRA | RMS LoRA | TEDS LoRA | Nota base |
|:---|:---:|:---:|:---:|:---:|:---:|
| ≤ 5 linhas | 88 | 29.7 | 10.7 | 48.7 | 23.3 |
| 6–20 linhas | 72 | 43.4 | 22.3 | 64.6 | 24.0 |
| > 20 linhas | 7 | 16.1 | 7.8 | 24.4 | 13.4 |

### Gráfico × diagrama
| Tipo | Painéis | Nota LoRA | RMS LoRA | TEDS LoRA | Nota base |
|:---|:---:|:---:|:---:|:---:|:---:|
| gráfico | 124 | 43.9 | 21.0 | 66.8 | 28.6 |
| diagrama / imagem (não é gráfico) | 43 | 9.6 | 0.0 | 19.3 | 7.5 |

### Por tipo de figura
| Tipo de figura (≥ 5 painéis) | Painéis | Nota LoRA | RMS LoRA | TEDS LoRA | Nota base |
|:---|:---:|:---:|:---:|:---:|:---:|
| multiple line chart | 43 | 38.1 | 20.1 | 56.2 | 15.2 |
| line chart | 29 | 50.8 | 26.9 | 74.8 | 39.8 |
| molecular structure diagram | 27 | 4.5 | 0.0 | 8.9 | 3.6 |
| scatter plot | 16 | 62.2 | 37.6 | 86.8 | 55.0 |
| multiple scatter plot | 14 | 34.7 | 10.4 | 59.0 | 22.8 |
| image panel | 7 | 10.8 | 0.0 | 21.6 | 10.1 |
| reaction scheme | 7 | 24.1 | 0.0 | 48.2 | 22.1 |
| stacked spectra chart | 7 | 32.9 | 0.0 | 65.8 | 18.8 |
| multi spectra chart | 6 | 32.3 | 2.2 | 62.5 | 8.6 |
| multi-axis chart | 5 | 36.3 | 10.8 | 61.9 | 29.7 |

---

## 🧪 Gabarito suspeito

A regra marcou **47 painéis (28.1%)**: vírgula decimal, `~`, intervalos, texto em coluna numérica ou mais de 50 linhas. Eles tiram **35.0** contra 35.1 dos demais, ou seja, **a mesma nota**. Então a regra ainda não separa os casos "impossíveis" e provavelmente está larga demais.

---

## ⚖️ Diferença para a análise do 7B

| | Análise do 7B (Lucas) | Esta análise |
| :--- | :--- | :--- |
| Métrica | local do repo (`icdar_score`) | **oficial** da competição |
| Unidade | figura inteira | **painel** (como o oficial) |
| Conjunto | teste (373) | **dev** (167 painéis) |
| Modelo | 7B + LoRA | 3B + LoRA e 3B base |
| Casos | piores casos, olhados à mão | todos os painéis, por regra automática |

As categorias de erro partiram do diagnóstico do 7B, e os padrões batem (grade inventada, painel omitido, cabeçalho, escala).

---

## 🔁 Como reproduzir

```bash
.venv/bin/python analyze_errors.py --metadata data/raw/validation_metadata.jsonl --repo-predictions outputs/analise_erros/lora3b/predictions.jsonl --out-dir outputs/analise_erros/lora3b
```

O `panels.csv` gerado tem uma linha por painel, com nota, categorias e tamanho das tabelas. O plano completo está em `docs/plano_analise_erros.md`.

---

## ⏭️ Próximos passos

1. **Decidir com o grupo** se os painéis que não são gráfico (diagramas com gabarito de texto) entram na nota que usamos para comparar modelos.
2. **Oráculo:** consertar por código o `~` e a escala e medir quanto de nota isso recupera.
3. Rodar o mesmo `analyze_errors.py` nas predições do 7B e dos experimentos novos (sintético, outros modelos).

---

## 📂 Arquivos

- [`README.md`](./README.md): este índice.
- [`01_diagrama_sem_dados_6_paineis.md`](./01_diagrama_sem_dados_6_paineis.md) + imagem
- [`02_tendencia_invertida.md`](./02_tendencia_invertida.md) + imagem
- [`03_series_numa_celula.md`](./03_series_numa_celula.md) + imagem
- [`04_rampa_inventada_bbox_trocado.md`](./04_rampa_inventada_bbox_trocado.md) + imagem
- [`05_eixo_categorico_sem_rotulo.md`](./05_eixo_categorico_sem_rotulo.md) + imagem
- [`06_valor_constante_gabarito_errado.md`](./06_valor_constante_gabarito_errado.md) + imagem
- [`07_gabarito_com_aproximacoes.md`](./07_gabarito_com_aproximacoes.md) + imagem
- [`08_metrica_generosa.md`](./08_metrica_generosa.md) + imagem
