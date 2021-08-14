# ocrdocker

对图像ocr进行识别，根据识别结果生成特定的json格式。

1，识别框架和预训练模型使用paddle ocr；

2，支持多目录、深层次目录、中文目录；

3，打包docker优化。



在项目下创建目录 /inference，将预识别模型放置该位置下。

ch_ppocr_mobile_v2.0_cls_infer

ch_ppocr_server_v2.0_det_infer

ch_ppocr_server_v2.0_rec_infer
