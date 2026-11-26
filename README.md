# Detecção de Ronco (YAMNet) — Biologix

Classificador **snore / not_snore** por evento para as amostras de 80 s (.m4a)
do exame domiciliar de sono da Biologix. Pipeline: detecção de eventos por
energia (Otsu) → embeddings YAMNet (congelado) → features v6 → XGBoost/SVM/
LogReg/MLP. Avaliação por paciente (GroupShuffleSplit) e por época de 5 s.

Histórico completo (v1–v6, testes que falharam, decisões, pendências):
documento **"Detecção de Ronco — Handoff do Projeto"**
(https://claude.ai/artifact/398fdcv79nJPrUuzCqFjPL).

> ⚠️ **Dados de paciente (LGPD).** Este repositório contém **só código**.
> Áudios, rótulos e embeddings ficam no Google Drive da Biologix, com acesso
> restrito. Não commite `.m4a`, `.csv`, `.npz` (o `.gitignore` já bloqueia).
> Mantenha o repositório **privado**.

---

## 1. O que está no Git e o que está no Drive

| Item | Onde | Por quê |
|---|---|---|
| Scripts `.py`, README, requirements | **Git** | Código, pequeno, versionável |
| `M4A_audios/` (áudios dos pacientes) | **Drive** | Dado sensível, pesado, fonte primária |
| `M4A_classification/` (rótulos manuais + embeddings) | **Drive** (zip datado) | Dado sensível; é o ativo mais caro (horas de escuta) |
| `M4A_classification_v5_frames/` | **Drive** (opcional) | Regenerável com `extractFrameEmbeddings.py` (~demora) |
| Modelos `.pkl` (quando existirem) | **Drive** | Binário, derivado dos dados |

Por que os rótulos **não** vão para o Git: cada `reviewed_<id>.csv` tem o código
do paciente + 1024 colunas de embedding (MBs por arquivo); uma vez commitado,
fica para sempre no histórico, mesmo que seja apagado depois.

### Estrutura no Drive (pasta `Snore Detection - dados`)

```
Snore Detection - dados/
├── LEIAME.txt                                   ← aponta para este repositório
├── M4A_audios.zip                               ← atualizar quando chegarem pacientes novos
├── rotulos/
│   ├── M4A_classification_2026-11-25.zip        ← backup datado; NUNCA sobrescrever
│   └── M4A_classification_AAAA-MM-DD.zip
└── frames/
    └── M4A_classification_v5_frames_2026-11-25.zip   ← opcional (regenerável)
```

Link da pasta: **`<COLE AQUI O LINK>`** (pedir acesso a Diego Domingues).

---

## 2. Como montar o ambiente do zero

1. **Clonar o código**
   ```
   git clone https://github.com/<org>/<repo>.git snorePythonM4A
   ```
2. **Criar o ambiente** (Anaconda Prompt)
   ```
   conda create -n audioSono python=3.11
   conda activate audioSono
   pip install -r requirements.txt
   conda install -c conda-forge ffmpeg
   ```
   O `ffmpeg` é obrigatório: o `pydub` usa ele para ler `.m4a`.
   (Se existir `requirements-lock.txt`, prefira `pip install -r requirements-lock.txt`
   para reproduzir as versões exatas dos resultados publicados.)
3. **Baixar os dados do Drive** e extrair **ao lado** da pasta do código:
   ```
   snore2026/                         ← DATA_ROOT
   ├── snorePythonM4A/                ← este repositório
   ├── M4A_audios/<codigo_paciente>/*.m4a
   ├── M4A_classification/            reviewed_<id>.csv, reviewed_<id>_done.txt, ...
   └── M4A_classification_v5_frames/  frames_<id>.npz, frames_meta_<id>.csv
   ```
   - Do zip de rótulos, use **o mais recente** da pasta `rotulos/`.
   - Ao extrair, confira que não ficou uma pasta duplicada
     (`M4A_audios/M4A_audios/...`).
4. **Ajustar os caminhos** (próxima seção) se `snore2026` não estiver em
   `C:\Users\diego\Documents\Python\`.
5. **Verificar**
   ```
   python checkSetup.py
   ```
   Ele lista pacotes, ffmpeg e dados, e diz o que falta e de onde baixar.
6. Na primeira execução, o YAMNet é baixado do TF Hub (precisa de internet).

---

## 3. Caminhos (hardcoded — editar no topo de cada script)

Todos os scripts assumem `DATA_ROOT = C:\Users\diego\Documents\Python\snore2026`.
Se a sua pasta for outra, troque esse prefixo nas constantes abaixo:

| Script | Constantes |
|---|---|
| `checkSetup.py` | `DATA_ROOT`, `DRIVE_URL` |
| `reviewAndEmbed2.py` | `PATIENTS_ROOT`, `OUTPUT_DIR` |
| `extractFrameEmbeddings.py` | `PATIENTS_ROOT`, `REVIEWED_CSV_FOLDER`, `FRAMES_OUTPUT_DIR` |
| `trainNN_main.py` | `REVIEWED_CSV_FOLDER`, `FRAMES_DIR`, `TEST_PREDICTIONS_CSV` |
| `computeEpochMetrics.py` | importa de `trainNN_main.py` |
| `suggestPatients.py` | `PATIENTS_ROOT`, `REVIEWED_CSV_FOLDER` |
| `reviewErrors.py` | `REVIEWED_CSV_FOLDER`, `PATIENTS_ROOT`, `TEST_PREDICTIONS_CSV` |

---

## 4. Scripts e ordem de execução

```
reviewAndEmbed2.py  ──►  extractFrameEmbeddings.py  ──►  trainNN_main.py  ──►  computeEpochMetrics.py
 (rotular + embedding)    (embeddings por frame)          (treino/CV v6)        (métrica por época 5 s)
        ▲                                                       │
        └──── suggestPatients.py (qual paciente ouvir)          └──► reviewErrors.py (ouvir/corrigir erros)
```

| Script | Faz | Lê | Escreve |
|---|---|---|---|
| `reviewAndEmbed2.py` | Detecta eventos, toca cada um, rótulo manual (`s`/`n`/`r`/`p`/`q`, `S`/`N` = resto do arquivo), calcula embedding YAMNet; mostra palpite do modelo atual | `M4A_audios` | `reviewed_<id>.csv`, `reviewed_<id>_done.txt` |
| `extractFrameEmbeddings.py` | Recalcula embeddings **por frame** dos eventos já rotulados (pula pacientes completos) | áudios + reviewed | `frames_<id>.npz`, `frames_meta_<id>.csv` |
| `trainNN_main.py` | Pipeline v6 (1038 features, 2 estágios), CV por paciente (`N_SPLITS=5`), média ± dp | reviewed + frames | `test_predictions_v6.csv` |
| `computeEpochMetrics.py` | Avaliação por época de 5 s (crédito ao filtro do detector); usa só arquivos `_done` | idem | console |
| `suggestPatients.py` | Ranqueia pacientes não rotulados por % estimado de not_snore e incerteza | áudios + reviewed | `patient_suggestions.csv` |
| `reviewErrors.py` | Toca eventos errados no teste; `c` corrige o rótulo no CSV | predições + reviewed | `reviewed_<id>.csv` |
| `energy.py`, `threshold.py`, `event.py`, `autoCorrelation.py`, `findPeaks.py`, `snoreConstants.py` | Módulos da detecção de eventos (importados) | — | — |

**Depois de rotular pacientes novos:** rode `extractFrameEmbeddings.py` antes
de treinar, senão os pacientes novos ficam sem features de frame.

---

## 5. Backup dos rótulos (rotina)

Ao terminar uma sessão de rotulagem relevante:

1. Compacte `M4A_classification/` → `M4A_classification_AAAA-MM-DD.zip`.
2. Suba em `Snore Detection - dados/rotulos/` **sem apagar** os anteriores.
3. (Opcional) Faça o mesmo com `M4A_classification_v5_frames/` em `frames/`.

---

## 6. Cuidados conhecidos

- **Nunca abra e salve os CSVs no Excel pt-BR**: ele troca `.` por `,` e
  corrompe as colunas numéricas. Para olhar, use pandas ou abra sem salvar.
- **Não deixe CSV externo (ESC-50) dentro de `M4A_classification/`**: o treino
  lê todo `reviewed_*.csv`, e o ESC-50 piorou o modelo. O `checkSetup.py` avisa.
- Arquivos `.m4a` sem nenhum evento detectado não são marcados como `_done`
  (bug conhecido; ver handoff).
- `reviewErrors.py` ainda aponta para `test_predictions_logreg.csv`;
  troque para `test_predictions_v6.csv` para revisar os erros do modelo atual.

---

## 7. Experimentos autônomos (Claude Code)

`runExperiment.py` roda uma configuração JSON em validação cruzada por paciente
**só nos pacientes de desenvolvimento**; os pacientes de `experiments/holdout.json`
ficam trancados para um único teste final. Protocolo completo (regras, métrica,
critério de parada, backlog): **`CLAUDE.md`**.

```
python runExperiment.py experiments/configs/exp_000_baseline.json          # um experimento
python runExperiment.py experiments/configs/exp_000_baseline.json --repeats 3
python runExperiment.py --compare exp_000_baseline_r3 exp_007_r3            # comparação pareada
python runExperiment.py --leaderboard
python runExperiment.py experiments/configs/<melhor>.json --final --i-am-diego   # holdout, 1 vez
```

- `experiments/holdout.json` tem códigos de paciente: fica fora do Git (backup no Drive junto dos rótulos).
- Os números do harness **não** são comparáveis com os do `trainNN_main.py`
  (o harness exclui o holdout do CV).
