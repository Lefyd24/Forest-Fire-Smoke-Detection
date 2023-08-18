This is a demo project designed to demonstrate the use of 2 trained models for inference - a YOLOv8 model and a YOLOv7 one, both trained on the same dataset, with the same resources, number of epochs and rest characteristics.<br>
The dataset used for training is the [Fire Image Dataset V2](https://universe.roboflow.com/kirzone/fire-iejes/dataset/2#) from [Roboflow](https://universe.roboflow.com/) (on open source website) and the models were trained on a Google Colab instance with a Tesla P100 GPU.<br>
<b>Model Training Params:</b>
| Type (Param)       	| Value 	|
|--------------------	|-------	|
| TOTAL IMAGES       	| 1706  	|
| EPOCHS             	| 30    	|
| IMAGE SIZE         	| 640   	|
| BATCH SIZE         	| 20    	|
| LR (Learning Rate) 	| 0.01  	|

Google Colab Notebooks used for training:
- [YOLOv8](https://colab.research.google.com/drive/1oOhKRR0QGHGdBYt3ru9HHZj8VXdTlAv3) (To be Replaced)
- [YOLOv7](https://colab.research.google.com/drive/1lWRhfprK58WxoUX5I38y3GW6-1rCWfzp#scrollTo=GD9gUQpaBxNa)
 