from transformers import MLukeTokenizer, LukeModel
import sentencepiece as spm
import torch
import csv
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

# 既存モデルの読み込み
def recommend(query,sentence_embeddings):
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
        sentences.append(query)
        query_embedding_vector = model.encode([query], batch_size=8)
        # print("query:")
        # print( query_embedding_vector.shape)
        sentence_embeddings = torch.vstack((sentence_embeddings, query_embedding_vector))
        print(sentence_embeddings.shape)
        # ビールの説明文、受け取った文章をエンコード（ベクトル表現に変換）
        # sentence_embeddings = model.encode(sentences, batch_size=8)
        # print(sentence_embeddings.shape)

        # 類似度上位1つを出力
        closest_n = 1

        distances = scipy.spatial.distance.cdist(
            [sentence_embeddings[-1]], sentence_embeddings, metric="cosine"
        )[0]

        results = zip(range(len(distances)), distances)
        results = sorted(results, key=lambda x: x[1])

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nオススメのゲームは:")

        index1= data[data['detail']==sentences[results[1][0]]].index[0]
        index2= data[data['detail']==sentences[results[2][0]]].index[0]

        return data.iloc[index1,1],data.iloc[index1,2],data.iloc[index1,3],data.iloc[index1,4],data.iloc[index1,5],data.iloc[index1,7],data.iloc[index1,8],data.iloc[index1,9],data.iloc[index1,10]

def read_csv():
     with open('gamedata4_df.csv', 'r', encoding='utf-8') as csvfile:
          reader = csv.DictReader(csvfile)
          data = list(reader)
     return data

# def search_timedata(csvfilename,targettime):
#      results = []

#      def extract_duration_range(duration):
#      # '~'が含まれている場合は時間範囲を取得
#           if '~' in duration:
#                return [int(value) for value in duration.replace('分', '').split('~')]
#           else:
#                # '~'が含まれていない場合はそのままの値を返す
#                return [int(duration.replace('分', ''))]
     
#      def time_in_range(time, time_range):
#      # 目標の時間を整数に変換
#           time = int(time)

#           if len(time_range) == 1:
#                if time == time_range[0]:
#                     return True

#           elif len(time_range) == 2:
#                if time >= time_range[0] and time <= time_range[1]:
#                     return True

#           else:
#                return False
#      def search_data_by_time(csv_filename, time):

#      # CSVファイルからデータを読み込む
#         with open(csv_filename, 'r', encoding='utf-8') as csvfile:
#           reader = csv.DictReader(csvfile)
          
#           # CSVデータをループして条件に一致するデータを探す
#           for row in reader:
#                time_range = extract_duration_range(row['time'])
#                print(time_range)
#                if time_in_range(time,time_range):
#                     results.append(row['title'])

#      return results

def search_data_by_duration(csv_filename, target_duration):
    results = []

    def extract_duration_range(duration):
        if '~' in duration:
            return [int(value) for value in duration.replace('分', '').split('~')]
        else:
            return [int(duration.replace('分', ''))]

    def is_duration_in_range(target_duration, duration_range):
        target_duration = int(target_duration)

        if len(duration_range) == 1:
            return target_duration == duration_range[0]
        elif len(duration_range) == 2:
            return target_duration >= duration_range[0] and target_duration <= duration_range[1]
        else:
            return False

    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            duration_range = extract_duration_range(row['time'])
            if is_duration_in_range(target_duration, duration_range):
                result_dict = {key: row[key] for key in ['title','few','people','time','install','tag','apple_url','google_url','web_url']}
                results.append(result_dict)

    return results

def search_data_by_people(csv_filename, target_people):
    results = []

    def extract_people_range(people):
        if '~' in people:
            return [int(value) for value in people.replace('人', '').split('~')]
        else:
            return [int(people.replace('人', ''))]

    def is_people_in_range(target_people, people_range):
        target_people = int(target_people)

        if len(people_range) == 1:
            return target_people == people_range[0]
        elif len(people_range) == 2:
            return target_people >= people_range[0] and target_people <= people_range[1]
        else:
            return False

    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        for row in reader:
            people_range = extract_people_range(row['people'])
            if is_people_in_range(target_people, people_range):
                result_dict = {key: row[key] for key in ['title','apple_url','google_url','web_url']}
                results.append(result_dict)

    return results
 
