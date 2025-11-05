# treino_ensemble_lemmas.py
import os, sys, json, joblib, numpy as np, pandas as pd
from typing import List, Tuple, Dict

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score, auc

from sentence_transformers import SentenceTransformer

CSV_PATH = "dados_lemmatized.csv"
MODEL_BIN = "modelo_ensemble_lemmas.joblib"
REPORT_JSON = "relatorio_ensemble_lemmas.json"
EMB_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SEED = 42

def carregar_dados(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"Ficheiro não encontrado: {path}"); sys.exit(1)
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    if not {"lemmas","label"}.issubset(df.columns):
        print("O CSV precisa de colunas lemmas e label"); sys.exit(1)

    if df["label"].dtype == "O":
        mapa = {"hate":1,"ódio":1,"odio":1,"ofensivo":1,"toxico":1,"toxic":1,
                "non-hate":0,"neutral":0,"neutro":0,"positivo":0,"normal":0,"nao-odio":0,"não-ódio":0}
        df["label"] = df["label"].map(lambda x: mapa.get(str(x).strip().lower(), np.nan))

    df = df.dropna(subset=["lemmas","label"])
    df["lemmas"] = df["lemmas"].astype(str).str.strip()
    df = df[df["lemmas"] != ""]
    df["label"] = df["label"].astype(int)

    print("Distribuição de classes:")
    print(df["label"].value_counts().sort_index())
    return df

def escolher_limiar(y_true: np.ndarray, p1: np.ndarray) -> Tuple[float, Dict[str,float]]:
    pr, rc, thr = precision_recall_curve(y_true, p1)
    f1s = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
    best_idx = np.nanargmax(f1s[:-1]) if len(f1s) > 1 else 0
    best_thr = float(thr[best_idx]) if len(thr) > 0 else 0.5
    info = {"best_f1": float(f1s[best_idx]) if len(f1s)>0 else 0.0,
            "best_precision": float(pr[best_idx]) if len(pr)>0 else 0.0,
            "best_recall": float(rc[best_idx]) if len(rc)>0 else 0.0,
            "pr_auc": float(auc(rc, pr)) if len(rc)>1 else 0.0}
    return best_thr, info

def treinar_componentes(X_tr, y_tr, X_va, y_va):
    # Modelo A: TF-IDF palavra n-gramas + LogReg
    pipe_word = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="word", ngram_range=(1,3), min_df=1, max_df=0.95)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.5, solver="liblinear", random_state=SEED))
    ])
    pipe_word.fit(X_tr, y_tr)

    # Modelo B: TF-IDF caracteres n-gramas + LinearSVC calibrada
    pipe_char = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=1, max_df=0.95)),
        ("svm", LinearSVC(C=1.0, random_state=SEED))
    ])
    # Calibração para obter probabilidades
    pipe_char.fit(X_tr, y_tr)
    # criar embeddings de saídas para calibrar com features fixas
    # calibramos com CalibratedClassifierCV diretamente sobre features TF-IDF
    tfidf_char = pipe_char.named_steps["tfidf"]
    svm_char = pipe_char.named_steps["svm"]
    X_tr_char = tfidf_char.transform(X_tr)
    X_va_char = tfidf_char.transform(X_va)
    svm_cal = CalibratedClassifierCV(svm_char, method="sigmoid", cv=5)
    svm_cal.fit(X_tr_char, y_tr)

    # Modelo C: embeddings em lemas + LogReg calibrada
    emb_model = SentenceTransformer(EMB_MODEL_NAME)
    E_tr = emb_model.encode(list(X_tr), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    E_va = emb_model.encode(list(X_va), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    lr_emb = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, solver="liblinear", random_state=SEED)
    lr_emb_cal = CalibratedClassifierCV(lr_emb, method="sigmoid", cv=5)
    lr_emb_cal.fit(E_tr, y_tr)

    # probabilidades de validação
    pA = pipe_word.predict_proba(X_va)[:,1]
    pB = svm_cal.predict_proba(X_va_char)[:,1]
    pC = lr_emb_cal.predict_proba(E_va)[:,1]

    # procurar pesos simples que maximizam F1 em validação
    melhor = {"f1": -1, "w": (1.0, 1.0, 1.0)}
    for wA in [0.5, 1.0, 1.5, 2.0]:
        for wB in [0.5, 1.0, 1.5, 2.0]:
            for wC in [0.5, 1.0, 1.5]:
                p_mix = (wA*pA + wB*pB + wC*pC) / (wA + wB + wC)
                thr, _ = escolher_limiar(y_va, p_mix)
                y_pred = (p_mix >= thr).astype(int)
                f1 = f1_score(y_va, y_pred, zero_division=0)
                if f1 > melhor["f1"]:
                    melhor = {"f1": f1, "w": (wA, wB, wC), "thr": thr}
    print(f"Pesos escolhidos A,B,C: {melhor['w']}  limiar ótimo: {melhor['thr']:.3f}  F1 validação: {melhor['f1']:.3f}")

    artefactos = {
        "pipe_word": pipe_word,
        "tfidf_char": tfidf_char,
        "svm_cal": svm_cal,
        "emb_model_name": EMB_MODEL_NAME,
        "lr_emb_cal": lr_emb_cal,
        "weights": melhor["w"],
        "threshold": float(melhor["thr"])
    }
    return artefactos

def avaliar(art: dict, X_te, y_te):
    # probabilidades dos três modelos
    pA = art["pipe_word"].predict_proba(X_te)[:,1]

    X_te_char = art["tfidf_char"].transform(X_te)
    pB = art["svm_cal"].predict_proba(X_te_char)[:,1]

    emb = SentenceTransformer(art["emb_model_name"])
    E_te = emb.encode(list(X_te), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    pC = art["lr_emb_cal"].predict_proba(E_te)[:,1]

    wA, wB, wC = art["weights"]
    p_mix = (wA*pA + wB*pB + wC*pC) / (wA + wB + wC)

    # avaliar com 0.50 e com limiar ótimo aprendido
    y_pred_050 = (p_mix >= 0.5).astype(int)
    y_pred_opt = (p_mix >= art["threshold"]).astype(int)

    print("\nRelatório teste com limiar 0.50")
    print(classification_report(y_te, y_pred_050, digits=3))
    print("Matriz 0.50:")
    print(confusion_matrix(y_te, y_pred_050))

    print("\nRelatório teste com limiar ótimo")
    print(classification_report(y_te, y_pred_opt, digits=3))
    print("Matriz ótimo:")
    print(confusion_matrix(y_te, y_pred_opt))

    return p_mix

def prever(art: dict, frases_lemmas: List[str]) -> pd.DataFrame:
    pA = art["pipe_word"].predict_proba(frases_lemmas)[:,1]
    X_char = art["tfidf_char"].transform(frases_lemmas)
    pB = art["svm_cal"].predict_proba(X_char)[:,1]
    emb = SentenceTransformer(art["emb_model_name"])
    E = emb.encode(list(frases_lemmas), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    pC = art["lr_emb_cal"].predict_proba(E)[:,1]

    wA, wB, wC = art["weights"]
    p_mix = (wA*pA + wB*pB + wC*pC) / (wA + wB + wC)
    y050 = (p_mix >= 0.5).astype(int)
    yopt = (p_mix >= art["threshold"]).astype(int)
    return pd.DataFrame({
        "lemmas": frases_lemmas,
        "p_odio": p_mix,
        "classe_050": y050,
        "classe_opt": yopt,
        "limiar_opt": art["threshold"]
    })

def main():
    df = carregar_dados(CSV_PATH)
    X = df["lemmas"].tolist()
    y = df["label"].to_numpy()

    # split treino e teste
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    # dentro do treino, separar validação para afinar pesos e limiar
    X_tr_in, X_va, y_tr_in, y_va = train_test_split(X_tr, y_tr, test_size=0.25, random_state=SEED, stratify=y_tr)
    print(f"Tamanho treino interno: {len(X_tr_in)}  validação: {len(X_va)}  teste: {len(X_te)}")

    art = treinar_componentes(X_tr_in, y_tr_in, X_va, y_va)
    _ = avaliar(art, X_te, y_te)

    joblib.dump(art, MODEL_BIN)
    print(f"\nModelo ensemble guardado em {MODEL_BIN}")

    # relatório simples
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump({"weights": art["weights"], "threshold": art["threshold"]}, f, ensure_ascii=False, indent=2)

    # exemplo prático
    exemplos = [
        "minoria doente impor doença mental",
        "respeitar pessoa valor fundamental sociedade",
        "não querer ver grupox televisão"
    ]
    print("\nExemplos de previsão:")
    print(prever(art, exemplos).to_string(index=False))

if __name__ == "__main__":
    main()
