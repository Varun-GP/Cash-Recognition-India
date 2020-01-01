import os

def RUN(train_file, save_path, data_directory):
	print(train_file,save_path,data_directory)

	os.system('sudo python '+ train_file+'   --bottleneck_dir= '+save_path+'   --how_many_training_steps 3000   --model_dir='+save_path+'   --output_graph='+save_path+'--output_labels='+save_path+' --image_dir '+data_directory)
