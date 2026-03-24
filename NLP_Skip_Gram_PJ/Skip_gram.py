import numpy as np
import random
import re
from pyvi import ViTokenizer
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd

def preprocess_text_vi(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = ViTokenizer.tokenize(text).split()
    stop_words_vi = {'của', 'là', 'và', 'theo', 'để', 'một', 'đã', 'có', 'trong', 'này', 'với', 'rằng', 
                    'còn', 'vẫn', 'như', 'không', 'từ', 'được', 'cho', 'nên', 'đó', 'đã', 'là'}
    tokens = [word for word in tokens if word not in stop_words_vi]
    return tokens

class Skip_Gram_Model:
    def __init__(self, text, window_size=2, d=15):
        tokens = preprocess_text_vi(text)
        self.cap_tu = self.taoCapTu(tokens,window_size)
        
        self.vocab = Counter(tokens)
        self.vocab_size = len(tokens)
        self.wordToIndex = {word: idx for idx, word in enumerate(self.vocab)}
        self.indexToWord = {idx: word for word, idx in self.wordToIndex.items()}

        self.W_in = np.random.randn(self.vocab_size, d)
        self.W_out = np.random.randn(d, self.vocab_size)

    def taoCapTu(self, tokens, window_size):    
        cap_tu = []
        for i in range(window_size, len(tokens) - window_size):
            target_word = tokens[i]
            context_word = tokens[i - window_size:i] + tokens[i + 1:i + window_size + 1]
            for word in context_word:
                cap_tu.append((target_word, word))
        return cap_tu
    
    def softmax(self,u):
        e_u = np.exp(u - np.max(u))
        return e_u / (e_u.sum(axis=0) + 1e-10)
    
    def forward(self, target_word_index):
        h = self.W_in[target_word_index]
        u = np.dot(h, self.W_out)
        y_pre = self.softmax(u)
        return y_pre
    
    def backward(self,target_word_index,y, learning_rate):
        y_pre = self.forward(target_word_index)
        h = self.W_in[target_word_index]
        error = y_pre - y

        dW_out = np.outer(h, error)
        self.W_out -= learning_rate * dW_out

        dh = np.dot(self.W_out, error.T)
        self.W_in[target_word_index] -= learning_rate * dh

    def train(self, learning_rate, epochs=100, batch_size=1200):
        lich_su_cua_ham_mat_mat = []
        for epoch in range(epochs):
            total_loss = 0
            random.shuffle(self.cap_tu)

            for i in range(0, len(self.cap_tu), batch_size):
                batch = self.cap_tu[i:i + batch_size]
                total_batch_loss = 0
                for target, context in batch:
                    target_word_index = self.wordToIndex[target]
                    context_word_index = self.wordToIndex[context]

                    y = np.zeros(self.vocab_size)
                    y[context_word_index] = 1

                    y_pred = self.forward(target_word_index)
                    loss = -np.sum(y * np.log(y_pred))
                    total_batch_loss += loss
                    self.backward(target_word_index, y, learning_rate)

                total_loss += total_batch_loss / batch_size

            total_loss /= len(self.cap_tu) / batch_size
            lich_su_cua_ham_mat_mat.append(total_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss}")

        return lich_su_cua_ham_mat_mat


    def predict(self, word):
        word_index = self.wordToIndex.get(word, None)
        if word_index is not None:
            return self.W_in[word_index]
        else:
            print(f"Từ '{word}' không nằm trong từ điển.")
            return None 

# Hàm tính toán cosine similarity
def tinhCosineSimilarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

# Hàm đánh giá tương đồng giữa hai từ
def doTuongDongGiuaHaiTu(word1, word2, model):
    vec1 = model.predict(word1)
    vec2 = model.predict(word2)
    if vec1 is not None and vec2 is not None:
        similarity = tinhCosineSimilarity(vec1, vec2)
        print(f"Tương đồng cosine giữa '{word1}' và '{word2}': {similarity}")

text = ""
with open('C:/Users/Admin/Desktop/XLNNTN/chuong_024.txt',encoding='utf-8') as f:
    text = f.read()

model = Skip_Gram_Model(text)
learning_rate = 0.01
loss_history = model.train(learning_rate)
plt.plot(loss_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Hàm Mất Mát Khi Huấn Luyện Ngữ Liệu')
plt.show()

word1 = "bệ_hạ"
y_pre1 = model.predict(word1)
print(f"Vector nhúng của từ {word1} là: {y_pre1}")

word1 = "nhà_vua"
y_pre1 = model.predict(word1)
print(f"Vector nhúng của từ {word1} là: {y_pre1}")

word1 = "tướng"
y_pre1 = model.predict(word1)
print(f"Vector nhúng của từ {word1} là: {y_pre1}")

doTuongDongGiuaHaiTu('nhà_vua', 'bệ_hạ', model)
doTuongDongGiuaHaiTu('nhà_vua', 'tướng', model)