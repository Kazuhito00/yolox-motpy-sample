# yolox-motpy-sample

[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)と[motpy](https://github.com/wmuron/motpy)を用いたMOT(Multiple Object Tracking)のPythonサンプルです。<br>
YOLOXは[YOLOX-ONNX-TFLite-Sample](https://github.com/Kazuhito00/YOLOX-ONNX-TFLite-Sample)で、ONNXに変換したモデルを使用しています。<br>

https://user-images.githubusercontent.com/37477845/153742550-19c51648-f355-40e2-ac7d-85f5809cc6b1.mp4

# Requirement 
* OpenCV 3.4.2 or later
* onnxruntime 1.5.2 or later

# Demo
デモの実行方法は以下です。
```bash
python sample.py
```
* --device<br>
カメラデバイス番号の指定<br>
デフォルト：0
* --movie<br>
動画ファイルの指定 ※指定時はカメラデバイスより優先<br>
デフォルト：指定なし
* --width<br>
カメラキャプチャ時の横幅<br>
デフォルト：960
* --height<br>
カメラキャプチャ時の縦幅<br>
デフォルト：540
<details>
<summary>YOLOXパラメータ</summary>
  
* --yolox_model<br>
ロードするモデルの格納パス<br>
デフォルト：model/yolox_nano.onnx
* --input_shape<br>
モデルの入力サイズ<br>
デフォルト：416,416
* --score_th<br>
クラス判別の閾値<br>
デフォルト：0.3
* --nms_th<br>
NMSの閾値<br>
デフォルト：0.45
* --nms_score_th<br>
NMSのスコア閾値<br>
デフォルト：0.1
* --with_p6<br>
Large P6モデルを使用するか否か<br>
デフォルト：指定なし
</details>

<details>
<summary>motpyパラメータ</summary>
  
* --max_staleness<br>
デフォルト：5
* --order_pos<br>
デフォルト：1
* --dim_pos<br>
デフォルト：2
* --order_size<br>
デフォルト：0
* --dim_size<br>
デフォルト：2
* --q_var_pos<br>
デフォルト：5000.0
* --r_var_pos<br>
デフォルト：0.1
* --tracker_min_iou<br>
デフォルト：0.25
* --multi_match_min_iou<br>
デフォルト：0.93
* --min_steps_alive<br>
デフォルト：3
  
※パラメータ詳細は[motpy](https://github.com/wmuron/motpy)を参照ください。
</details>


# Reference
* [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [YOLOX-ONNX-TFLite-Sample](https://github.com/Kazuhito00/YOLOX-ONNX-TFLite-Sample)
* [motpy](https://github.com/wmuron/motpy)

# Author
高橋かずひと(https://twitter.com/KzhtTkhs)
 
# License 
yolox-motpy-sample is under [MIT License](LICENSE).

# License(Movie)
サンプル動画は[NHKクリエイティブ・ライブラリー](https://www.nhk.or.jp/archives/creative/)の[ケニア共和国キツイ 町並み(4) ふかんショット](https://www2.nhk.or.jp/archives/creative/material/view.cgi?m=D0002040395_00000)を使用しています。
