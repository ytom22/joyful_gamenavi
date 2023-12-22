from transformers import MLukeTokenizer, LukeModel
import sentencepiece as spm
import torch
import scipy.spatial
import pandas as pd

class SentenceLukeJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = MLukeTokenizer.from_pretrained(model_name_or_path)
        self.model = LukeModel.from_pretrained(model_name_or_path)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    @torch.no_grad()
    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx : batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(
                batch, padding="longest", truncation=True, return_tensors="pt"
            ).to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(
                model_output, encoded_input["attention_mask"]
            ).to("cpu")

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)



import json

MODEL_NAME = "sonoisa/sentence-luke-japanese-base-lite"
model = SentenceLukeJapanese(MODEL_NAME)

# 説明文を入れるリストを作成
sentences = []

# CSVファイルのパスを指定
csv_file_path = 'gamedata4_df.csv'

# 読み込む列の名前を指定
target_column_name = 'detail'

# CSVファイルをDataFrameとして読み込む
data = pd.read_csv(csv_file_path)

# 指定した列のデータをリストに追加
sentences = data[target_column_name].tolist()

# 標準入力で、理想のビールのイメージを文章で受け取る
# query = input()
# query = "aaa"
# sentences.append(query)

# ビールの説明文、受け取った文章をエンコード（ベクトル表現に変換）
sentence_embeddings = model.encode(sentences, batch_size=8)
print(sentence_embeddings.shape)
torch.save(sentence_embeddings,"sentence_embeddings.pt")

# for vector in sentence_embeddings:
#     print(vector)

# with open("embedding_vecotrs.json","w",encoding = "utf-8"):
#     json.dump()
        
