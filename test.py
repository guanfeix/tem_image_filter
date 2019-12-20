import csv
import json
csv_file = '/Users/zy/Desktop/query_result_wb.csv'
html_file = '/Users/zy/Desktop/query_result_wb.html'
text_head = """<!DOCTYPE html>
<html>
<head>
<style>
.flex-container {
  display: flex;
  flex-wrap: wrap;
  background-color: DodgerBlue;
}

.flex-container > div {
  background-color: #f1f1f1;
  width: 23.5%;
  margin: 10px;
  text-align: center;
  line-height: 15px;
  font-size: 15px;
}
.flex-item {
 position: relative;
}
.img {
  width: 300px;
 height: 300px;
}
.label {
  position: absolute;
  bottom: -5%;
  left: 50%;
  transform: translate(-50%, -50%);
  background-color: white; 
}
</style>
</head>
<body>
<div class="flex-container">
"""
text ="""  <div class="flex-item">
  <a href="{}" target="_blank">
  	<image class="img" src="{}"></image>
  	<p class="label">{}</p>
  </a>
  </div>
"""
text_tail = """</div>
</body>
</html>"""
all_results = []
with open(csv_file, encoding='UTF-8-sig') as f:
    csvReader = csv.DictReader(f)
    for content in csvReader:
        # print(content)
        url = content['pic_oss_url']
        raw_json = content['raw_json']
        data = eval(raw_json)
        result = data['filter_result']
        filter_tags = data['filter_tags']
        reason = filter_tags.get('base_tags')
        a=text.format(url,url,str(result)+str(reason))
        all_results.append(a)
        print(a,content)

all_results = ''.join(all_results)
text = text_head+all_results+text_tail
with open(html_file, 'w') as f:
    f.write(text)
    print('suc')