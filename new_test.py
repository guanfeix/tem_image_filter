import csv
import json

csv_file = '/Users/zy/Desktop/query_result_ins.csv'
html_file_ok = '/Users/zy/Desktop/ins_ok.html'
html_file_fail = '/Users/zy/Desktop/ins_fail.html'
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
all_results_ok = []
all_results_fail = []
if __name__ == '__main__':

    with open(csv_file, encoding='UTF-8-sig') as f:
        csvReader = csv.DictReader(f)
        for content in csvReader:
            print(content)
            url = content['pic_oss_url']
            raw_json = content['raw_json']
            # raw_json.replace(r'\\', '')
            # print(raw_json)
            # data = json.loads(raw_json)

            data = eval(raw_json)
            result = data['filter_result']
            result_v2 = data['filter_result_V2']
            filter_tags = data['filter_tags']
            reason = filter_tags.get('base_tags')
            item = text.format(url,url,'V1:'+str(result)+'V2:'+str(result_v2)+str(reason))
            if result:
                all_results_ok.append(item)
            else:
                all_results_fail.append(item)
            print(item, content)

    all_results_ok = ''.join(all_results_ok)
    all_results_fail = ''.join(all_results_fail)
    text_ok = text_head+all_results_ok+text_tail
    text_fail = text_head+all_results_fail+text_tail
    with open(html_file_ok, 'w') as f:
        f.write(text_ok)
        print('suc--ok')

    with open(html_file_fail, 'w') as f:
        f.write(text_fail)
        print('suc--fail')