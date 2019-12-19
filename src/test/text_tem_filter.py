import csv
import requests
csv_file = ''
all_results = []
with open(csv_file, encoding='UTF-8-sig') as f:
    csvReader = csv.DictReader(f)
    for content in csvReader:
        url = content['pic_oss_url']
        all_results.append(url)

for url in all_results:
    data = {}
    data['imageUrl'] = url
    data['image_source'] = 'ins'
    requests.post(url, data)
