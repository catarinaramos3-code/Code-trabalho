import re, html, csv, unicodedata
import collections, collections.abc
if not hasattr(collections, "Sequence"): collections.Sequence = collections.abc.Sequence

import monkeypatch_nltk
from NLPyPort.FullPipeline import new_full_pipe

# spaCy opcional para verbos no infinitivo
try:
    import spacy
    _SPACY = spacy.load("pt_core_news_sm")
except Exception:
    _SPACY = None

# ========= Config =========
ENABLE_DOMAIN_FIXES = True  # põe False se não quiseres normalizações de nomes próprios

_PT_STOPWORDS = {
    "a","o","os","as","um","uma","uns","umas","de","do","dos","da","das","d","em","no","na","nos","nas","num","numa","nuns","numas",
    "para","por","pelo","pela","pelos","pelas","e","ou","mas","nem","que","porque","como","se","quando","onde","enquanto",
    "eu","tu","ele","ela","nós","vos","eles","elas","me","te","se","nos","vos","lhe","lhes",
    "meu","minha","meus","minhas","teu","tua","teus","tuas","seu","sua","seus","suas",
    "este","esta","estes","estas","esse","essa","esses","essas","aquele","aquela","aqueles","aquelas",
    "isto","isso","aquilo","ser","estar","ter","haver","é","era","foi","vai","vão","são"
}

# Ortografia AO90 mínima
_AO90_MAP = {
    "reflectir":"refletir","reflecte":"reflete","perspectiva":"perspetiva","óptimo":"ótimo","acção":"ação"
}

# CORREÇÃO MANUAL MELHORADA: Normaliza logo para o singular onde possível
_FIX_MAP = {
    "eleiçõe":"eleição",     # Eleições -> Eleição (Singularizado)
    "opiniõe":"opinião",     # Opiniões -> Opinião (Singularizado)
    "individuo":"indivíduo",
    "méro":"mérito",
    "perspetivo":"perspetiva",
    "vivemo":"viver",        # Vivemos -> Viver (Infinitivo)
    "exérco":"exército",
    "aparce":"aparecer",     # Aparce -> Aparecer (Infinitivo)
    "aberraçõe":"aberração", # Aberrações -> Aberração (Singularizado)
    "seboso": "seboso",
    "cromo": "cromo"
}

# CORREÇÃO DE DOMÍNIO MELHORADA: Mais normalizações de termos específicos
_DOMAIN_FIXES = {
    "milene":"milei",
    "put":"putin",
    "tram":"trans",
    "transformer": "trans",       # Adicionado: 'transformer' -> 'trans'
    "paneleiragem": "paneleiro",  # Adicionado
    "paneleirice": "paneleiro",   # Adicionado
    "wokismo": "woke"             # Adicionado: Normalização de estrangeirismo
}

# Formas verbais comuns -> infinitivo (fallback mínimo se spaCy falhar)
_VERB_FALLBACK = {
    "acharei":"achar","acharia":"achar","achava":"achar","achou":"achar","acho":"achar",
    "constrói":"construir","construí":"construir","construo":"construir",
    "têm":"ter","tenho":"ter","tens":"ter","tem":"ter","tivemos":"ter","tiveram":"ter",
    "dê":"dar","dei":"dar","dás":"dar","dá":"dar","damos":"dar","deram":"dar",
    "vê":"ver","vejo":"ver","vês":"ver","vêem":"ver","vi":"ver","viram":"ver",
    "há":"haver","havia":"haver","houve":"haver","hão":"haver",
    "vai":"ir","vou":"ir","vais":"ir","vamos":"ir","vão":"ir",
    "vem":"vir","venho":"vir","véns":"vir","vêm":"vir"
}

# Tokens lixo a descartar
_JUNK_TOKENS = {"``","''","...","..",":","«","»","“","”","‘","’","xy","el","fdp","fdx","caralho"} # 'fdp', 'fdx', 'caralho' adicionados

def _norm_unicode(s: str) -> str:
    return unicodedata.normalize("NFC", s)

def limpar(texto: str) -> str:
    texto = _norm_unicode(html.unescape(texto))
    texto = re.sub(r"http\S+|www\.\S+", " ", texto)
    texto = re.sub(r"<[^>]+>", " ", texto)
    texto = re.sub(r"[\u200B-\u200D\uFEFF]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def _norm_pos(pos: str) -> str:
    up = pos.upper()
    if up.startswith("PROP") or "PROPN" in up: return "PROPN"
    if up.startswith("N"): return "NOUN"
    if up.startswith("V"): return "VERB"
    if "ADJ" in up: return "ADJ"
    if up.startswith("ADV") or "ADV" in up: return "ADV"
    return up

def _fix_ortografia(lemma: str, ao90: bool) -> str:
    if not ao90: return lemma
    if lemma in _AO90_MAP: return _AO90_MAP[lemma]
    # reparar finais comuns cortados
    lemma = re.sub(r"coes$", "ções", lemma)   # eleicoes -> eleições
    lemma = re.sub(r"çoe$", "ções", lemma)    # eleiçoe -> eleições
    lemma = re.sub(r"cao$", "ção", lemma)     # acao -> ação
    return lemma

# FUNÇÃO DE SINGULARIZAÇÃO MELHORADA
def _plural_to_singular(lemma: str) -> str:
    # Plurais irregulares ou mais complexos:
    if lemma.endswith("ões"): return lemma[:-3] + "ão"  # opiniões -> opinião
    if lemma.endswith("ães"): return lemma[:-3] + "ão"  # pães -> pão
    if lemma.endswith("ais"): return lemma[:-3] + "al"  # animais -> animal
    if lemma.endswith("eis"): return lemma[:-3] + "el"  # papéis -> papel
    if lemma.endswith("ns"): 
        if lemma.endswith("gens"): return lemma[:-2] # viagens -> viagem
        return lemma[:-2] + "m"                   # bens -> bem
    # -is de -il tónico (fáceis -> fácil), evitar países/raiz
    if lemma.endswith("is") and not re.search(r"[áéíóú]is$", lemma):
        return lemma[:-2] + "il"
    
    # Plurais simples (-s):
    if lemma.endswith("s") and len(lemma) > 2 and lemma not in ("os", "as", "dos", "das"):
        # Se for -es, tenta remover. (Ex: flores -> flor)
        if lemma.endswith("es"):
            return lemma[:-2] 
        # Se for -s simples (Ex: gatos -> gato)
        return lemma[:-1] 
        
    return lemma

def _looks_broken(token: str, lemma: str) -> bool:
    t, l = _norm_unicode(token), _norm_unicode(lemma)
    vowels = set("aeiouáéíóúâêôãõ")
    if len(l) < 3 and len(t) >= 4: return True
    if sum(ch in vowels for ch in l.lower()) == 0 and sum(ch in vowels for ch in t.lower()) >= 2: return True
    if abs(len(t) - len(l)) >= max(3, len(t)//2): return True
    return False

# FALLBACK DE INFINITIVO MELHORADO (mais validação com spaCy)
def _spacy_infinitivo_fallback(token: str, lemma: str) -> str:
    # usa mapa rápido antes de spaCy (mais barato)
    l = _VERB_FALLBACK.get(lemma, None)
    if l: return l
    
    if _SPACY is None: return lemma
    
    # Tenta spaCy se o lemma não parece infinitivo ou é uma forma irregular comum
    if not re.search(r"(ar|er|ir)$", lemma) or lemma in _VERB_FALLBACK:
        doc = _SPACY(token)
        if len(doc) == 1:
            cand = doc[0].lemma_
            # Garantir que o spaCy o classificou como VERBO e que o lema é um infinitivo
            if _norm_pos(doc[0].pos_) == "VERB" and re.search(r"(ar|er|ir)$", cand):
                return cand
            # Tentar se o lema do spaCy é diferente e mais longo (bom para formas muito reduzidas)
            elif cand != token and len(cand) > len(lemma) and re.search(r"(ar|er|ir)$", cand):
                return cand

    return lemma

def lematizar_frases(
    frases,
    guardar_csv: str | None = None,
    allowed_pos: set[str] | None = None,
    remove_stopwords: bool = False,
    extra_stopwords: set[str] | None = None,
    ortografia_ao90: bool = True,
    corrigir_plurais: bool = True,
    usar_spacy_fallback: bool = True
):
    frases_limpas = [limpar(f) for f in frases]

    opts = {
        "tokenizer": True, "pos_tagger": True, "lemmatizer": True,
        "entity_recognition": False, "np_chunking": False, "string_or_array": True
    }
    # Usa o NLPyPort
    doc = new_full_pipe(frases_limpas, options=opts)

    resultados = []
    atual = []
    for tok, pos, lem in zip(doc.tokens, doc.pos_tags, doc.lemas):
        if tok == "EOS":
            if atual: resultados.append(atual)
            atual = []
            continue

        tok_n = _norm_unicode(tok)
        lem_n = _norm_unicode(lem)

        # corrige lemas “estragados”
        if _looks_broken(tok_n, lem_n):
            lem_n = tok_n.lower()

        # fixes manuais e de domínio
        lem_n = _FIX_MAP.get(lem_n, lem_n)
        if ENABLE_DOMAIN_FIXES:
            lem_n = _DOMAIN_FIXES.get(lem_n, lem_n)

        # AO90 antes de singularizar
        lem_n = _fix_ortografia(lem_n, ortografia_ao90)

        # singularização
        if corrigir_plurais:
            lem_n = _plural_to_singular(lem_n)

        # verbos -> infinitivo (com o fallback melhorado)
        if usar_spacy_fallback and _norm_pos(pos) == "VERB":
            lem_n = _spacy_infinitivo_fallback(tok_n, lem_n)

        # filtrar lixo e tokens sem letras ou muito curtos
        if (tok_n.lower() in _JUNK_TOKENS) or (not re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]", tok_n)) or (len(tok_n) < 2 and not tok_n.isdigit()):
            continue
        
        # Filtrar lemas muito curtos após lematização se não forem números e não estiverem no vocab de fixes
        if len(lem_n) < 2 and not lem_n.isdigit() and lem_n not in _FIX_MAP.values() and lem_n not in _DOMAIN_FIXES.values():
             continue

        atual.append((tok_n, pos, lem_n))

    if atual:
        resultados.append(atual)

    def _passa(pos: str, lem: str) -> bool:
        posn = _norm_pos(pos)
        if allowed_pos and posn not in allowed_pos:
            return False
        if remove_stopwords:
            sw = set(_PT_STOPWORDS)
            if extra_stopwords: sw |= {w.lower() for w in extra_stopwords}
            if lem.lower() in sw:
                return False
        return True

    filtrados = []
    for sent in resultados:
        sent_f = [(tok, pos, lem) for tok, pos, lem in sent if _passa(pos, lem)]
        filtrados.append(sent_f)

    if guardar_csv:
        with open(guardar_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["frase_id", "ord", "token", "pos", "lema"])
            for i, sent in enumerate(filtrados, start=1):
                for j, (tok, pos, lem) in enumerate(sent, start=1):
                    w.writerow([i, j, tok, pos, lem])

    lemas_por_frase = [[lem for _, _, lem in sent] for sent in filtrados]
    return filtrados, lemas_por_frase
