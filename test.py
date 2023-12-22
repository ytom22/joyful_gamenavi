import csv

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
                results.append(row['title'])

    return results

# 使用例
csv_filename = 'gamedata4_df.csv'
target_duration_to_search = 15  # 目標の時間（例: 15分）
matching_titles = search_data_by_duration(csv_filename, target_duration_to_search)

# 結果を出力
print(f"検索条件 {target_duration_to_search} に一致するデータ:")
for title in matching_titles:
    print(title)
