import os
import torch
import math
import csv
from flask import (
     Flask, 
     request, 
     render_template)
from model_test import (
     recommend,
     search_data_by_duration,
     search_data_by_people) #model.pyからrecommend関数をインポート

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST']) #トップページのルーティング
def index():
    return render_template('index.html')

@app.route('/search', methods=['GET'])   #ワード検索ページのルーティング
def search():
    return render_template('search.html')

@app.route('/search2', methods=['GET'])  #タグ検索ページ
def search2():
    return render_template('search2.html')

@app.route('/result', methods=['GET', 'POST'])   #ワード検索→出力ページのルーティング
def result():
    if request.method == "GET":
        return render_template('result.html')
    elif request.method == "POST":
     
        sentence_embeddings = torch.load("sentence_embeddings.pt")
        try:

            fav = request.form.get("fav")# name属性がfavのtextボックスから単一の値を取得
           
            title, few, people, time, install, detail, apple_url, google_url, web_url = recommend(fav,sentence_embeddings)
            # if math.isnan(apple_url):
            #     apple_url = None
            if isinstance(apple_url, float) and math.isnan(apple_url):
                apple_url = None
            if isinstance(google_url, float) and math.isnan(google_url):
                google_url = None
            if isinstance(web_url, float) and math.isnan(web_url):
                web_url = None
            return render_template('result.html', title=title, few=few, people=people, time=time, install=install, detail=detail, apple_url=apple_url, google_url=google_url, web_url=web_url)#左辺がHTML、右辺がPython側の変数
        except KeyError as e:
            return f"KeyError: {e}"
        
@app.route('/result2', methods=['GET', 'POST'])   #タグ検索→出力ページ2のルーティング
def result2():
    if request.method == "GET":
        return render_template('tab_result.html')
    elif request.method == "POST":
        try:
            # time = request.form.getlist("time")# name属性がfavのtextボックスから単一の値を取得
            # people = request.form.getlist("people")

            # print(time)
            # print(people)

            
            # # results = search_data_by_duration("gamedata4_df.csv",time)
            # # results = search_data_by_people("gamedata4_df.csv",people)
            # if time:
            #     results = search_data_by_duration("gamedata4_df.csv", time[0])
            # elif people:
            #     results = search_data_by_people("gamedata4_df.csv", people[0])

            time = request.form.getlist("time")# name属性がfavのtextボックスから単一の値を取得
            # results = [game['title'] for game in data if game['Duration'] == fav]            
            results=[]
            results = search_data_by_duration("gamedata4_df.csv", time[0])
            return render_template('tab_result.html', results=results)#左辺がHTML、右辺がPython側の変数
        except KeyError as e:
            return f"KeyError: {e}"


            

if __name__ == "__main__":
    app.run(debug=True)