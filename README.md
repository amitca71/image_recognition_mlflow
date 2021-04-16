prerequisite: installed mlflow (pip install mlflow)

executing the all flow: mlflow run .

load_raw_data: 
         - download file from google driver and unzip
etl_data: 
         - split to train and validation dir
train_keras: 
		- data augmentation
		- hyper parameter tuning
		- log class labeling
		- log model
		
json inference: 
		- deploy the model (localhost)
		
sample client usage: curl -X POST -F image=@./artist_dataset/frida_kahlo/Without-Hope-1945-Frida-Kahlo.jpg 'http://localhost:5000/predict'

		



