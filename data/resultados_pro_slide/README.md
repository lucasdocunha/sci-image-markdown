# Resultados da Avaliação Oficial (Task 2 - Sci-ImageMiner)

Avaliação realizada no split de **teste oficial (373 figuras, 605 painéis)** utilizando a métrica oficial por painel (DePlot Datapoints F1 para RMS e PubTables-1M para TEDS).

### Referências da Competição no Teste:
- **1º Lugar do Leaderboard:** 41,81
- **Baseline Oficial:** 35,97

### Tabela Comparativa:

| Checkpoint | Métrica Local (antigo) | Oficial (`weighted`) | RMS Oficial | TEDS Oficial | Painéis Avaliados | Tabelas Extras | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| LightOnOCR-3-4B (Recorde) | 52,32 | 38.18 | 15.73 | 60.62 | 605 | 29 | Concluído |
| LightOnOCR-3-4B + Synth15 | 51,72 | 37.30 | 14.91 | 59.69 | 605 | 69 | Concluído |
| Qwen2.5-VL-7B-Instruct | 52,11 | 36.15 | 13.46 | 58.85 | 605 | 117 | Concluído |
| Qwen3-VL-8B-Instruct | 48,72 | 32.39 | 12.18 | 52.61 | 605 | 36 | Concluído |
| Qwen2.5-VL-3B-Instruct (Orig Res) | 49,49 | 34.83 | 9.98 | 59.68 | 605 | 214 | Concluído |

---
*Gerado automaticamente pelo pipeline de avaliação oficial.*