# CLAUDE.md — Protocolo de melhoria autônoma do classificador de ronco

Você (Claude Code) vai rodar experimentos sozinho para melhorar o classificador
snore / not_snore. Leia este arquivo inteiro antes de começar e siga-o à risca.
Contexto do projeto: `README.md`. Histórico v1–v6: documento "Detecção de
Ronco — Handoff do Projeto" (link no README).

Ambiente: Windows, conda env `audioSono`. Rode tudo com `python` dentro desse
env, a partir desta pasta (`snorePythonM4A`).

---

## 1. Regras invioláveis

1. **Nunca use os pacientes do holdout.** Eles estão em `experiments/holdout.json`.
   Não edite esse arquivo, não leia os dados desses pacientes por fora do
   harness, não rode `runExperiment.py --final`. O teste final é do Diego.
2. **Nunca modifique dados**: nada em `..\M4A_audios`, `..\M4A_classification`,
   `..\M4A_classification_v5_frames`. Nada de corrigir rótulos, mover ou apagar
   CSVs. Se achar um problema nos dados, anote no relatório.
3. **Não mude a régua**: em `runExperiment.py`, não altere `PRIMARY_METRIC`,
   `CV_SEED`, `N_SPLITS`, `TEST_SIZE`, `CLIP_DURATION_SEC`, `epoch_pairs`,
   `binary_metrics`, nem a lógica de remoção do holdout. Se achar que a régua
   está errada, pare e explique no relatório.
4. **Nunca sobrescreva nem apague resultados** em `experiments/results/` ou
   `experiments/experiments_log.csv`.
5. **Git**: trabalhe na branch `experimentos` (crie se não existir). Pode
   commitar código, configs e `experiments/results/*.json`. **Nunca faça push.**
   Nunca commite dados (o `.gitignore` bloqueia; confira `git status`).
6. Não instale pacotes sem perguntar. Não rode `reviewAndEmbed2.py`,
   `reviewErrors.py` nem nada que toque áudio ou abra janela.

## 2. Objetivo e métrica

- **Métrica primária:** `epoch_bal_acc` = (sensibilidade + especificidade)/2
  por época de 5 s, média da validação cruzada por paciente nos pacientes de
  desenvolvimento (todos menos o holdout). O harness calcula.
- Reporte sempre também: `epoch_sens`, `epoch_esp`, `event_f1`, `event_auc`.
- Um experimento só é **promovido** (vira o novo melhor) se, na comparação
  pareada com o melhor atual (`--compare`), com `--repeats 3` nos dois:
  Δ médio ≥ +0.005 **e** ganhar em ≥ 70% dos splits. O comando imprime SIM/NÃO.
- Desconfie de ganho que vem só de um lado: se `epoch_sens` cair mais de 0.03
  em relação ao melhor atual, não promova sem registrar o trade-off.

## 3. Loop

```
0. git checkout -b experimentos   (ou checkout se já existir)
1. python runExperiment.py experiments/configs/exp_000_baseline.json
   python runExperiment.py experiments/configs/exp_000_baseline.json --repeats 3
   -> melhor atual = exp_000_baseline_r3
2. Repita:
   a. Escolha a próxima ideia do backlog (seção 4) ou uma derivada dos resultados.
   b. Escreva experiments/configs/exp_NNN_<nome>.json (NNN sequencial, nunca reutilize).
   c. Triagem: python runExperiment.py <config>            (repeats=1, rápido)
   d. Se a triagem ficar acima do melhor atual em repeats=1 (comparar com
      o baseline r1 via --compare), confirme: --repeats 3 e
      python runExperiment.py --compare <melhor_atual_r3> <exp_NNN_r3>
   e. Se SIM -> promova (anote em experiments/NOTAS.md).
   f. Anote em experiments/NOTAS.md uma linha: id, hipótese, resultado, decisão.
   g. Commit a cada ~5 experimentos: configs + results + NOTAS.md.
3. Pare quando atingir o critério de parada (seção 5) e escreva o relatório.
```

Se os dados mudarem no meio (o `data_sig` no log muda porque o Diego rotulou
mais pacientes), **rode o baseline de novo** com novo id
(`exp_NNN_baseline_rerun`) e compare só experimentos com o mesmo `data_sig`.

### Estendendo o harness

Pode adicionar novas opções ao `runExperiment.py` (ex.: PCA, pós-processamento)
desde que:
- a opção nova seja um campo do JSON com default que **não muda** o
  comportamento atual;
- depois da mudança você rode o baseline com um id novo e confirme que
  `primary_mean` é idêntico ao anterior (mesmo `data_sig`). Se não for, desfaça;
- respeite a regra 3.

## 4. Backlog de ideias (ordem sugerida)

Configs disponíveis hoje (ver docstring de `runExperiment.py`):

1. **Limiar.** Cada resultado tem `best_sweep_thr`. Teste o baseline com o
   limiar sugerido (ex.: `"threshold": 0.4`).
2. **`"frame_oof": true`** — hoje as features de vizinhança/meta do treino usam
   probabilidades de frame dentro da amostra (otimistas), enquanto as do teste
   são fora da amostra. OOF corrige o descasamento. Prioridade alta.
3. **XGBoost**: `scale_pos_weight` (≈ n_not_snore / n_snore), `max_depth` 3–8,
   `learning_rate` 0.02–0.1 com `n_estimators` ajustado, `subsample` 0.7–0.9,
   `colsample_bytree` 0.1–0.5 (são 1024+ features), `min_child_weight`.
4. **SVM**: `C` em {0.3, 1, 3, 10}, `gamma` em {"scale", 1e-4, 3e-4, 1e-3}.
5. **LogReg** com `C` baixo (0.01–0.3): às vezes ganha em dado pequeno.
6. **Ablação de blocos** (`"features"`): sem `emb`; sem `neigh`; sem
   `frame_meta`; só `emb`. Mostra o que carrega o resultado (útil para o artigo).
7. **Janela de vizinhança**: `neigh_past_sec`/`neigh_future_sec` em 5, 15, 20.
8. **Modelo de frame**: `frame_model_C` em {0.01, 0.1, 10}.
9. **Ensemble** (`"type": "ensemble"`) dos melhores XGBoost + SVM + LogReg.
10. **Calibração** (`"calibrate": true`) — combina com ajuste de limiar.

Precisam estender o harness (seção 3):

11. PCA nas 1024 dims de embedding antes do classificador (64/128/256 comps).
12. Pós-processamento temporal: suavizar probabilidades entre eventos vizinhos
    do mesmo arquivo antes de binarizar.
13. Features de contexto extras: posição do evento no clipe, número de eventos
    no arquivo, energia/duração relativas ao arquivo.

**Fora de escopo sem aprovação do Diego:** mudar a detecção de eventos
(`event.py`, `energy.py`, `threshold.py`), filtrar eventos do conjunto de
avaliação (ex.: megaeventos > 3 s), dados externos (ESC-50 já piorou),
fine-tuning do YAMNet.

## 5. Critério de parada

Pare no primeiro que acontecer:
- 40 execuções de `runExperiment.py` (contando triagem e confirmação), ou
- 10 ideias seguidas sem promoção, ou
- 6 horas de trabalho.

## 6. Relatório final — `experiments/RELATORIO.md`

1. Baseline vs melhor configuração: `epoch_bal_acc`, `epoch_sens`,
   `epoch_esp`, `event_f1`, `event_auc` (média ± dp, repeats=3), e a saída do
   `--compare`.
2. Config JSON do melhor experimento.
3. Tabela de todos os experimentos (`python runExperiment.py --leaderboard`).
4. O que funcionou, o que não funcionou, e hipóteses para os resultados.
5. Ablação de blocos (se feita).
6. Riscos: quantos experimentos foram testados (viés de seleção), variância
   entre splits, qualquer suspeita nos dados.
7. Recomendação: qual config o Diego deve rodar uma única vez no holdout:
   `python runExperiment.py experiments/configs/<melhor>.json --final --i-am-diego`

Commit final na branch `experimentos` com o relatório. Sem push.
