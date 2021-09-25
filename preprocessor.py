import numpy as np

from tqdm import tqdm


class SMILES_Tokenizer():  # 변수 설정
    def __init__(self, max_length):  # 생성자 생성 / 클래스의 생성자를 만들때 쓰는 항상 동일한 규칙
        self.txt2idx = {}  # selfs는 자기자신을 호출 / 파이썬에서는 클래스에서 사용하는 함수의 첫번째 인자(parameter)를 self로 사용하는 것이 원칙
        self.idx2txt = {}
        self.max_length = max_length

    def fit(self, SMILES_list):
        unique_char = set()  # 중복되지 않은 원소(unique)를 얻고자 할 때 쓰는 파이썬 내장 함수(sum, min, max와 같이 바로 사용 가능)
        for smiles in SMILES_list:
            for char in smiles:
                unique_char.add(char)  # set 선언하고 그 안에 계속 추가하고 싶으면 add() 사용
        unique_char = sorted(list(unique_char))
        for i, char in enumerate(unique_char):  # 순서가 있는 자료형(list, set, tuple, dictionary, string)을 입력 받아 순서와 리스트 값을 전달
            self.txt2idx[char] = i + 2
            self.idx2txt[i + 2] = char

    def txt2seq(self, texts):
        seqs = []
        for text in tqdm(texts):
            seq = [0] * self.max_length
            for i, t in enumerate(text):
                if i == self.max_length:
                    break
                try:
                    seq[i] = self.txt2idx[t]
                except:
                    seq[i] = 1
            seqs.append(seq)
        return np.array(seqs)