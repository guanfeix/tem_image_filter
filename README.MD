# 测试脚本
##### 本地测试-cpu:tensorflow==1.4.0

##### 服务器测试-gpu:tensorflow-gpu==1.4.0

## 测试
image/detect:
curl -w %{time_total} -X POST  -H "Content-Type:application/json"  http://gpu003:8766/image/detect/category -d 
'{"imageUrl": "https://zhiyi-image.oss-cn-hangzhou.aliyuncs.com/old_df/t51.2885-15/e35/25013009_2039369276305779_5433226615481958400_n.jpg","sizeFilterThres":"0.005","image_source":"ins"}'
image/filter:
## 旧版过滤器
curl -w %{time_total} -X POST  -H "Content-Type:application/json"  http://gpu003:8765/image/filter -d \
'{"imageUrl": "https://zhiyi-image.oss-cn-hangzhou.aliyuncs.com/data_infra/18/025815723e4036adbc1db3d297e5d082_1d3e7f7c929bf96e9b8b70572eacc82f.jpg","sizeFilterThres":"0.005","image_source":"ins"}'


curl -w %{time_total} -X POST  -H "Content-Type:application/json"  http://0.0.0.0:9003/image/filter -d \
'{"imageUrl": "https://zhiyi-image.oss-cn-hangzhou.aliyuncs.com/data_infra/18/a72ae84fe1b263727e3c86cb99bd9a0b_120c18d9b9bfbee777cf5e89ea16db32.jpg","sizeFilterThres":"0.005","image_source":"ins"}'


detect/recognition
curl -X POST http://gpu003:8768/image/detect/recognition -d  '{"imageUrl":"https://zhiyi-image.oss-cn-hangzhou.aliyuncs.com/old_df/instagram/927185_658783480854602_924916565_n.jpg","positions": [{"label":"上衣","score":"0.88577724","xmax":"0.65","xmin":"0.31","ymax":"0.78","ymin":"0.34"},{"label":"包","score":"0.5309918","xmax":"0.72","xmin":"0.49","ymax":"0.86","ymin":"0.59"}]}'