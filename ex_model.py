import csv

def extract_duration_range(duration):
    # '~'が含まれている場合は時間範囲を取得
    if '~' in duration:
        return [int(value) for value in duration.replace('分', '').split('~')]
    else:
        # '~'が含まれていない場合はそのままの値を返す
        return [int(duration.replace('分', ''))]
    
def time_in_range(time, time_range):
    # 目標の時間を整数に変換
     time = int(time)

    
     if len(time_range) == 1:
          if time == time_range[0]:
               return True

     elif len(time_range) == 2:
          if time >= time_range[0] and time <= time_range[1]:
               return True

     else:
          return False

def search_data_by_time(csv_filename, time):
    results = []

    # CSVファイルからデータを読み込む
    with open(csv_filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # CSVデータをループして条件に一致するデータを探す
        for row in reader:
            time_range = extract_duration_range(row['time'])
          #   print(time_range)
            if time_in_range(time,time_range):
                results.append(row['title'])

    return results

# 使用例
csv_filename = 'gamedata4_df.csv'
duration_to_search = '15'
matching_titles = search_data_by_time(csv_filename, duration_to_search)

# 結果を出力
print(f"検索条件 {duration_to_search} に一致するデータ:")
for title in matching_titles:
    print(title)
