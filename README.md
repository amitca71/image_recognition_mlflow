prerequisite: installed mlflow  version 1.15.0 (pip install mlflow)

executing the all flow: 
            mlflow run .


in order to see historical executions: 
           mlflow ui (open browser on: http://127.0.0.1:5000)

pipeline steps
load_raw_data: 
         - download file from google driver and unzip
etl_data: 
         - split to train and validation dir
train_keras: 
		- data augmentation
		- hyper parameter tuning
		- log class labeling
		- log model
		
model_inference: 
		- deploy the model (localhost)
		
sample client usage: curl -X POST -F image=@./artist_dataset/frida_kahlo/Without-Hope-1945-Frida-Kahlo.jpg 'http://localhost:5000/predict'

		
note: model training uses very small hard coded iterations related parameter, in order to be executed fast.
on real life, these parameters should be retrieved as input 


