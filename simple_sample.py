#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2

from yolox.yolox_onnx import YoloxONNX
from motpy import Detection, MultiObjectTracker


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    # YOLOX parameters
    parser.add_argument(
        "--yolox_model",
        type=str,
        default='model/yolox_nano.onnx',
    )
    parser.add_argument(
        '--input_shape',
        type=str,
        default="416,416",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.3,
        help='Class confidence',
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.45,
        help='NMS IoU threshold',
    )
    parser.add_argument(
        '--nms_score_th',
        type=float,
        default=0.1,
        help='NMS Score threshold',
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )

    # motpy parameters
    parser.add_argument(
        "--max_staleness",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--order_pos",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--dim_pos",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--order_size",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--dim_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--q_var_pos",
        type=float,
        default=5000.0,
    )
    parser.add_argument(
        "--r_var_pos",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--tracker_min_iou",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--multi_match_min_iou",
        type=float,
        default=0.93,
    )
    parser.add_argument(
        "--min_steps_alive",
        type=int,
        default=3,
    )

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.movie is not None:
        cap_device = args.movie

    # YOLOX parameters
    model_path = args.yolox_model
    input_shape = tuple(map(int, args.input_shape.split(',')))
    score_th = args.score_th
    nms_th = args.nms_th
    nms_score_th = args.nms_score_th
    with_p6 = args.with_p6

    # motpy parameters
    max_staleness = args.max_staleness
    order_pos = args.order_pos
    dim_pos = args.dim_pos
    order_size = args.order_size
    dim_size = args.dim_size
    q_var_pos = args.q_var_pos
    r_var_pos = args.r_var_pos
    tracker_min_iou = args.tracker_min_iou
    multi_match_min_iou = args.multi_match_min_iou
    min_steps_alive = args.min_steps_alive

    # カメラ準備 ###############################################################
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    # モデルロード #############################################################
    # Object Detection
    yolox = YoloxONNX(
        model_path=model_path,
        input_shape=input_shape,
        class_score_th=score_th,
        nms_th=nms_th,
        nms_score_th=nms_score_th,
        with_p6=with_p6,
        providers=['CPUExecutionProvider'],
    )

    # Multi Object Tracking
    tracker = MultiObjectTracker(
        dt=1 / cap_fps,
        tracker_kwargs={'max_staleness': max_staleness},
        model_spec={
            'order_pos': order_pos,
            'dim_pos': dim_pos,
            'order_size': order_size,
            'dim_size': dim_size,
            'q_var_pos': q_var_pos,
            'r_var_pos': r_var_pos
        },
        matching_fn_kwargs={
            'min_iou': tracker_min_iou,
            'multi_match_min_iou': multi_match_min_iou
        },
    )

    # COCOクラスリスト読み込み
    with open('coco_classes.txt', 'rt') as f:
        coco_classes = f.read().rstrip('\n').split('\n')

    # トラッキングID保持用変数
    track_id_dict = {}

    while True:
        start_time = time.time()

        # カメラキャプチャ ################################################
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # 推論実施 ########################################################
        # Object Detection
        bboxes, scores, class_ids = yolox.inference(frame)
        detections = [
            Detection(box=b, score=s, class_id=l)
            for b, s, l in zip(bboxes, scores, class_ids)
        ]

        # Multi Object Tracking
        _ = tracker.step(detections=detections)
        track_results = tracker.active_tracks(min_steps_alive=min_steps_alive)

        # トラッキングIDと連番の紐付け
        for track_result in track_results:
            if track_result.id not in track_id_dict:
                new_id = len(track_id_dict)
                track_id_dict[track_result.id] = new_id

        elapsed_time = time.time() - start_time

        # デバッグ描画
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            score_th,
            track_results,
            track_id_dict,
            coco_classes,
        )

        # キー処理(ESC：終了) ##############################################
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #########################################################
        cv2.imshow('YOLOX motpy Sample', debug_image)

    cap.release()
    cv2.destroyAllWindows()


def get_id_color(index):
    temp_index = abs(int(index)) * 3
    color = ((37 * temp_index) % 255, (17 * temp_index) % 255,
             (29 * temp_index) % 255)
    return color


def draw_debug(
    image,
    elapsed_time,
    score_th,
    track_results,
    track_id_dict,
    coco_classes,
):
    debug_image = copy.deepcopy(image)

    for track_result in track_results:
        tracker_id = track_id_dict[track_result.id]
        bbox = track_result.box
        class_id = int(track_result.class_id)
        score = track_result.score

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if score_th > score:
            continue

        color = get_id_color(tracker_id)

        # バウンディングボックス
        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            color,
            thickness=2,
        )

        # クラスID、スコア
        score = '%.2f' % score
        text = '%s:%s' % (str(coco_classes[int(class_id)]), score)
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            thickness=2,
        )

    # 推論時間
    text = 'Elapsed time:' + '%.0f' % (elapsed_time * 1000)
    text = text + 'ms'
    debug_image = cv2.putText(
        debug_image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )

    return debug_image


if __name__ == '__main__':
    main()
