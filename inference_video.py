#! /usr/bin/python3
# coding=utf-8

import torch
from model import MattingNetwork
from inference import convert_video


if __name__ == '__main__':

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	# option1: load mobilenetv3 model
	model = MattingNetwork('mobilenetv3').eval().to(device)
	model.load_state_dict(torch.load('work/checkpoint/rvm_mobilenetv3.pth'))

	# output video
	convert_video(
		model,                                       # 模型，可以加载到任何设备（cpu 或 cuda）
		input_source='work/video/input.avi',         # 视频文件，或图片序列文件夹
		# num_workers=1,                             # 只适用于图片序列输入，读取线程
		# input_resize=(1080, 720),                  # [可选项] 缩放视频大小
		output_type='video',                         # 可选 "video"（视频）或 "png_sequence"（PNG 序列）
		output_background='image',                   # [可选项] 定义输出视频或图片序列的背景, 默认"default", 可选 "green", "white", "image"
		output_composition='work/video/output.avi',  # 若导出视频，提供文件路径。若导出 PNG 序列，提供文件夹路径
		output_alpha="work/video/pha.avi",           # [可选项] 输出透明度预测
		output_foreground="work/video/fgr.avi",      # [可选项] 输出前景预测
		output_video_mbps=4,                         # 若导出视频，提供视频码率
		downsample_ratio=None,                       # 下采样比，可根据具体视频调节，或 None 自动下采样至 512px
		seq_chunk=1,                                 # 设置多帧并行计算
		progress=True,                               # 显示进度条
	)

	# output png_sequence
	convert_video(
		model,                                        # 模型，可以加载到任何设备（cpu 或 cuda）
		input_source='work/video/input.avi',          # 视频文件，或图片序列文件夹
		# num_workers=1,                              # 只适用于图片序列输入，读取线程
		# input_resize=(1080, 720),                   # [可选项] 缩放视频大小
		output_type='png_sequence',                   # 可选 "video"（视频）或 "png_sequence"（PNG 序列）
		output_background='image',                    # [可选项] 定义输出视频或图片序列的背景, 默认"default", 可选 "green", "white", "image"
		output_composition='work/video/output',       # 若导出视频，提供文件路径。若导出 PNG 序列，提供文件夹路径
		output_alpha="work/video/pha",                # [可选项] 输出透明度预测
		output_foreground="work/video/fgr",           # [可选项] 输出前景预测
		downsample_ratio=None,                        # 下采样比，可根据具体视频调节，或 None 自动下采样至 512px
		seq_chunk=1,                                  # 设置多帧并行计算
		progress=True                                 # 显示进度条
	)

