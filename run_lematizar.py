# lematizar_frases_csv.py
import pandas as pd
from lematizar import lematizar_frases  # importa a tua função

# === CONFIGURAÇÃO ===
INPUT_FILE = "frases.csv"     # ficheiro de entrada com as frases
OUTPUT_FILE = "lemmas.csv"    # ficheiro de saída com os lemas
TEXT_COLUMN = "texto"         # nome da coluna que contém as frases

# === LER O FICHEIRO ===
try:
    df = pd.read_csv(INPUT_FILE, encoding="utf-8")
except FileNotFoundError:
    raise SystemExit(f"❌ Ficheiro '{INPUT_FILE}' não encontrado. "
                     "Garante que está na mesma pasta do script.")
except Exception as e:
    raise SystemExit(f"Erro ao ler o ficheiro CSV: {e}")

if TEXT_COLUMN not in df.columns:
    raise SystemExit(f"❌ O CSV precisa de uma coluna chamada '{TEXT_COLUMN}'.")

# === CONVERTER EM LISTA DE FRASES ===
frases = df[TEXT_COLUMN].dropna().astype(str).tolist()

# === PARAMETROS DE LEMATIZAÇÃO ===
allowed = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
extra_stop = {"woke"}

# === EXECUTAR LEMATIZAÇÃO ===
res, lemas = lematizar_frases(
    frases,
    guardar_csv=OUTPUT_FILE,      # grava automaticamente em lemmas.csv
    allowed_pos=allowed,
    remove_stopwords=True,
    extra_stopwords=extra_stop
)

# === MOSTRAR RESULTADOS ===
for i, sent in enumerate(lemas, start=1):
    print(f"[lemas frase {i}] {sent}")

print(f"\n✅ Total de frases processadas: {len(frases)}")
print(f"📁 CSV gravado em '{OUTPUT_FILE}'")
