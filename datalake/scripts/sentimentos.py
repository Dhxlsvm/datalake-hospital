import pandas as pd
import re
import nltk
import unicodedata

# NLTK
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Carregar arquivo
df = pd.read_csv("comentarios_instagram.csv", sep=";", encoding="utf-8")
df["texto"] = df["texto"].astype(str)

# Stopwords
stop_words_pt = set(stopwords.words('portuguese'))

# Remover acentos
def remover_acentos(texto):
    return ''.join(
        c for c in unicodedata.normalize('NFKD', texto)
        if not unicodedata.combining(c)
    )

# Limpeza
def limpar_e_processar_texto(texto):
    texto = remover_acentos(str(texto).lower())
    texto = re.sub(r"http\S+|www\S+|https\S+", "", texto)
    texto = re.sub(r"@\w+|#", "", texto)
    texto = re.sub(r"[^a-zA-Z\s]", " ", texto)
    tokens = word_tokenize(texto, language="portuguese")
    return " ".join(tokens)

df["texto_processado"] = df["texto"].apply(limpar_e_processar_texto)

# Léxicos
lexico_positivo = {
    "bom", "boa", "otimo", "excelente", "maravilhoso", "maravilhosa",
    "perfeito", "parabens", "sensacional", "amei", "adorei",
    "incrivel", "top", "show", "legal", "que", "muito"
}

lexico_negativo = {
    "ruim", "pessimo", "horrivel", "demora", "demorado",
    "atraso", "atrasado", "falta", "erro", "espero",
    "esperando", "falha", "reclamar", "pior", "absurdo"
}

# Classificação
def classificar_sentimento(texto):
    score = 0
    for token in texto.split():
        if token in lexico_positivo:
            score += 1
        elif token in lexico_negativo:
            score -= 1
    return "Positivo" if score > 0 else "Negativo" if score < 0 else "Neutro"

df["Sentimento"] = df["texto_processado"].apply(classificar_sentimento)

# Exportar
df.to_csv("comentarios_tratados_sentimento.csv", index=False, encoding="utf-8")
