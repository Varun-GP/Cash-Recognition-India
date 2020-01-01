import os

def data_preparation(video_folder,save_path):
	videos_path = [x[2] for x in os.walk(video_folder)]
	videos_path = [item for sublist in videos_path for item in sublist]
	print(videos_path)
	print(save_path)

	for it in videos_path:
		if it.endswith('.mp4'):
			path = save_path + "/" + it.split('.mp4')[0]
			print(path)
			os.mkdir(path)
			os.system('ffmpeg -i '+video_folder+'/'+it+' -vf fps=1 '+path+'/1_%d.jpg')

