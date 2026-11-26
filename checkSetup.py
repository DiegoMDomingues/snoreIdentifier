"""
checkSetup.py
=============
Verifica se o ambiente e os dados estão prontos para rodar o pipeline de
detecção de ronco. Rode ANTES de qualquer outro script:

    python checkSetup.py

Se algo estiver faltando, o script diz o que é e de onde baixar.
Não altera nenhum arquivo.
"""

import glob
import importlib
import os
import shutil
import sys

# ---------------------------------------------------------------------------
# CONFIGURAÇÃO — ajuste DATA_ROOT para a pasta onde você colocou os dados.
# Os demais scripts usam os mesmos caminhos (ver README.md, "Caminhos").
# ---------------------------------------------------------------------------
DATA_ROOT = r"C:\Users\diego\Documents\Python\snore2026"
DRIVE_URL = "<COLE AQUI O LINK DA PASTA 'Snore Detection - dados' NO DRIVE DA BIOLOGIX>"

PATIENTS_ROOT       = os.path.join(DATA_ROOT, "M4A_audios")
REVIEWED_CSV_FOLDER = os.path.join(DATA_ROOT, "M4A_classification")
FRAMES_DIR          = os.path.join(DATA_ROOT, "M4A_classification_v5_frames")

PACKAGES = [
    ("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
    ("sklearn", "scikit-learn"), ("xgboost", "xgboost"),
    ("tensorflow", "tensorflow"), ("tensorflow_hub", "tensorflow-hub"),
    ("librosa", "librosa"), ("pydub", "pydub"), ("soundfile", "soundfile"),
    ("sounddevice", "sounddevice"), ("matplotlib", "matplotlib"),
]

ok_all = True


def report(ok, msg, fix=None):
    global ok_all
    print(("  [OK]    " if ok else "  [FALTA] ") + msg)
    if not ok:
        ok_all = False
        if fix:
            for line in fix.strip().splitlines():
                print("          -> " + line.strip())


print("\n1) Pacotes Python")
for mod, pip_name in PACKAGES:
    try:
        m = importlib.import_module(mod)
        report(True, f"{pip_name} {getattr(m, '__version__', '')}")
    except Exception as e:  # noqa: BLE001
        report(False, f"{pip_name} ({type(e).__name__})",
               f"pip install {pip_name}   (ou: pip install -r requirements.txt)")

print("\n2) ffmpeg (necessário para ler .m4a)")
report(shutil.which("ffmpeg") is not None, "ffmpeg no PATH",
       "conda install -c conda-forge ffmpeg   (no env ativo)\n"
       "ou baixe em https://ffmpeg.org e adicione a pasta bin ao PATH")

print(f"\n3) Dados (DATA_ROOT = {DATA_ROOT})")
if not os.path.isdir(DATA_ROOT):
    report(False, "DATA_ROOT não existe",
           "Crie a pasta ou ajuste DATA_ROOT neste arquivo e nos scripts (README, 'Caminhos').")

# 3a. Áudios
patients = [d for d in glob.glob(os.path.join(PATIENTS_ROOT, "*")) if os.path.isdir(d)]
n_m4a = len(glob.glob(os.path.join(PATIENTS_ROOT, "*", "*.m4a")))
report(len(patients) > 0 and n_m4a > 0,
       f"M4A_audios: {len(patients)} pacientes, {n_m4a} arquivos .m4a",
       f"Baixe 'M4A_audios.zip' do Drive: {DRIVE_URL}\n"
       f"Extraia para que fique {PATIENTS_ROOT}\\<codigo_paciente>\\*.m4a\n"
       "Necessário para: reviewAndEmbed2, extractFrameEmbeddings, reviewErrors, suggestPatients")

# 3b. Rótulos + embeddings
reviewed = [f for f in glob.glob(os.path.join(REVIEWED_CSV_FOLDER, "reviewed_*.csv"))]
done = glob.glob(os.path.join(REVIEWED_CSV_FOLDER, "reviewed_*_done.txt"))
report(len(reviewed) > 0,
       f"M4A_classification: {len(reviewed)} reviewed_*.csv, {len(done)} _done.txt",
       f"Baixe o zip MAIS RECENTE 'M4A_classification_AAAA-MM-DD.zip' do Drive: {DRIVE_URL}\n"
       f"Extraia para {REVIEWED_CSV_FOLDER}\n"
       "Necessário para: trainNN_main, computeEpochMetrics, extractFrameEmbeddings, reviewErrors.\n"
       "Sem ele, só dá para começar a rotular do zero com reviewAndEmbed2.")
external = [f for f in reviewed if "external" in os.path.basename(f).lower()]
if external:
    report(False, "CSV externo dentro de M4A_classification: "
           + ", ".join(os.path.basename(f) for f in external),
           "Mova para fora da pasta: o treino lê todo reviewed_*.csv e o ESC-50 piora o modelo.")

# 3c. Embeddings por frame
n_npz = len(glob.glob(os.path.join(FRAMES_DIR, "frames_*.npz")))
n_meta = len(glob.glob(os.path.join(FRAMES_DIR, "frames_meta_*.csv")))
report(n_npz > 0 and n_npz == n_meta,
       f"M4A_classification_v5_frames: {n_npz} .npz, {n_meta} meta .csv",
       "Regenerável: rode  python extractFrameEmbeddings.py  (precisa de M4A_audios + M4A_classification).\n"
       f"Ou baixe 'M4A_classification_v5_frames_AAAA-MM-DD.zip' do Drive: {DRIVE_URL}")
if reviewed and n_npz and n_npz < len(reviewed) - len(external):
    print("          (aviso) há pacientes rotulados sem frames — rode extractFrameEmbeddings.py")

print("\n" + ("Tudo pronto." if ok_all else "Há itens faltando (veja acima)."))
print("Obs.: na 1ª execução o YAMNet é baixado do TF Hub (precisa de internet).\n")
sys.exit(0 if ok_all else 1)
