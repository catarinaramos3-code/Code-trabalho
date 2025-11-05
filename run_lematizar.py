
import pandas as pd
from lematizar import lematizar_frases  

INPUT_FILE = "frases.csv"     
OUTPUT_FILE = "lemmas.csv"    
TEXT_COLUMN = "texto"        

try:
    df = pd.read_csv(INPUT_FILE, encoding="utf-8")
except FileNotFoundError:
    raise SystemExit(f"❌ Ficheiro '{INPUT_FILE}' não encontrado. "
                     "Garante que está na mesma pasta do script.")
except Exception as e:
    raise SystemExit(f"Erro ao ler o ficheiro CSV: {e}")

if TEXT_COLUMN not in df.columns:
    raise SystemExit(f"❌ O CSV precisa de uma coluna chamada '{TEXT_COLUMN}'.")

frases = df[TEXT_COLUMN].dropna().astype(str).tolist()

allowed = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
extra_stop = {"woke"}


res, lemas = lematizar_frases(
    frases,
    guardar_csv=OUTPUT_FILE,      
    allowed_pos=allowed,
    remove_stopwords=True,
    extra_stopwords=extra_stop
)

for i, sent in enumerate(lemas, start=1):
    print(f"[lemas frase {i}] {sent}")

print(f"\n✅ Total de frases processadas: {len(frases)}")
print(f"📁 CSV gravado em '{OUTPUT_FILE}'")
