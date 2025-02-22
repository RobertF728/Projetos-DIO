import numpy as np
import tkinter as tk
import nltk
nltk.download('punkt')  # Isso deve baixar o pacote punkt corretamente.
nltk.download('punkt_tab')


from tkinter import scrolledtext
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Baixar pacotes necessários do NLTK (se necessário)
nltk.download('punkt')

# Criar um stemmer para reduzir palavras ao seu radical
stemmer = PorterStemmer()

def preprocess_text(text):
    """Reduz as palavras ao seu radical para melhorar a correspondência."""
    tokens = word_tokenize(text.lower())  # Tokeniza e converte para minúsculas
    stemmed_tokens = [stemmer.stem(word) for word in tokens]  # Aplica stemming
    return " ".join(stemmed_tokens)

# Base de conhecimento expandida
perguntas = [
    "O que é um impedimento no futebol?",
    "Quando um gol é válido?",
    "O que acontece quando a bola sai pela linha lateral?",
    "Quando um jogador está impedido?",
    "Como funciona o tiro de meta?",
    "O que é um escanteio e quando ele é marcado?",
    "Como é cobrado um lateral?",
    "O que acontece quando a bola toca na mão de um jogador?",
    "Quando o goleiro pode pegar a bola com as mãos?",
    "O que é um pênalti e quando ele é marcado?",
    "Quantos jogadores podem estar em campo?",
    "Qual o tempo de duração de uma partida?",
    "O que acontece no empate de um jogo eliminatório?",
    "Quais motivos levam um jogador a receber um cartão amarelo?",
    "Quando um jogador recebe um cartão vermelho?",
    "O que é uma falta no futebol?",
    "Como funciona uma cobrança de falta direta e indireta?",
    "Se uma falta ocorre dentro da área, o que acontece?",
    "O que é um gol contra?",
    "Qual a função do VAR?",
    "O que acontece se um jogador cometer uma infração?",
    "O que define uma falta?",
    "Quais são os tipos de falta?",
    "Falta no futebol como funciona?",
    "Quais faltas geram cartão?",
    "Expulsão no futebol acontece como?",
]

respostas = [
    "Um jogador está impedido quando está mais próximo da linha do gol adversário do que a bola e o penúltimo defensor no momento em que recebe um passe.",
    "Um gol é válido quando a bola ultrapassa completamente a linha de gol entre as traves e abaixo do travessão.",
    "O time adversário ao que tocou na bola por último cobra um arremesso lateral, lançando a bola de volta ao campo com as mãos.",
    "Quando recebe um passe estando mais à frente do penúltimo defensor e interferindo na jogada.",
    "O tiro de meta é concedido quando a bola cruza completamente a linha de fundo, tendo sido tocada por um jogador do time atacante.",
    "O escanteio é concedido quando a bola cruza completamente a linha de fundo e foi tocada por último por um jogador do time defensor.",
    "O jogador deve lançar a bola com as duas mãos, por cima da cabeça, mantendo os pés no chão e fora do campo.",
    "Se o toque for intencional ou proporcionar vantagem ao jogador, é marcada uma falta.",
    "Dentro de sua área, quando a bola não for recuada por um passe intencional de um companheiro com os pés.",
    "O pênalti é uma penalidade cobrada a partir da marca do pênalti quando um jogador comete uma infração dentro de sua própria área.",
    "Cada time pode ter no máximo 11 jogadores em campo, incluindo o goleiro.",
    "Uma partida de futebol tem dois tempos de 45 minutos cada, com um intervalo de 15 minutos.",
    "Se houver empate em um jogo eliminatório, pode haver prorrogação de 30 minutos. Se o empate persistir, vai para os pênaltis.",
    "Cartão amarelo é dado por conduta antidesportiva, reclamação excessiva ou faltas reiteradas.",
    "Cartão vermelho ocorre por faltas graves, agressões ou dois amarelos na mesma partida.",
    "Uma falta acontece quando um jogador infringe as regras, como empurrar, segurar ou dar um carrinho perigoso.",
    "Uma cobrança direta pode ir ao gol, enquanto a indireta precisa ser tocada por outro jogador antes do chute.",
    "Se uma falta ocorre dentro da área do time defensor, é marcado um pênalti para o adversário.",
    "Um gol contra acontece quando um jogador acidentalmente chuta a bola para o próprio gol.",
    "O VAR revisa lances polêmicos para auxiliar os árbitros em decisões importantes.",
    "Se um jogador comete uma infração, o árbitro pode dar uma advertência, um cartão amarelo ou vermelho.",
    "Uma falta é definida por qualquer ação que interfira ilegalmente na jogada.",
    "Faltas podem ser técnicas (pé alto, carrinho) ou disciplinares (agressão, reclamação excessiva).",
    "Se um jogador comete uma falta grave ou reincide em faltas menores, pode ser punido com cartão.",
    "Jogadores podem ser expulsos por jogadas violentas, agressões, impedir um gol com as mãos ou receber dois amarelos.",
]

# Aplicar preprocessamento nas perguntas
perguntas_processadas = [preprocess_text(p) for p in perguntas]

# Vetorização usando TF-IDF
vectorizer = TfidfVectorizer()
perguntas_vetorizadas = vectorizer.fit_transform(perguntas_processadas)

def responder_pergunta(pergunta_usuario):
    """Processa e responde a pergunta do usuário."""
    pergunta_usuario = preprocess_text(pergunta_usuario)
    pergunta_usuario_vetorizada = vectorizer.transform([pergunta_usuario])
    similaridades = cosine_similarity(pergunta_usuario_vetorizada, perguntas_vetorizadas)
    indice_mais_similar = np.argmax(similaridades)

    if np.max(similaridades) < 0.2:
        return "Desculpe, não sei a resposta para isso."

    return respostas[indice_mais_similar]

# Criando interface gráfica
def enviar_pergunta():
    pergunta = entrada_pergunta.get()
    if pergunta.lower() == "sair":
        janela.quit()
    resposta = responder_pergunta(pergunta)
    chat_text.insert(tk.END, "Você: " + pergunta + "\n")
    chat_text.insert(tk.END, "Bot: " + resposta + "\n\n")
    entrada_pergunta.delete(0, tk.END)

# Criando a janela principal
janela = tk.Tk()
janela.title("ChatBot Futebol ⚽")
janela.geometry("500x400")

# Área de exibição do chat
chat_text = scrolledtext.ScrolledText(janela, wrap=tk.WORD, width=60, height=15)
chat_text.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

# Campo de entrada da pergunta
entrada_pergunta = tk.Entry(janela, width=50)
entrada_pergunta.grid(row=1, column=0, padx=10, pady=10)

# Botão de envio
botao_enviar = tk.Button(janela, text="Enviar", command=enviar_pergunta)
botao_enviar.grid(row=1, column=1, padx=10, pady=10)

# Executar a interface gráfica
janela.mainloop()
