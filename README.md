# Tensorflowlite to counting people with edge TPU (On coral dev)

Cummulative people counting Tensorflow Lite.

![Cumulative counting example](Doc/exemple.jpg)

## Installation
1. Clone the repository 
   ```git clone https://github.com/Lazorme/IAfinal.git```
   
   install [linux-image-4.19.125-mtk_4-2_arm64](linux-image-4.19.125-mtk_4-2_arm64.deb) (Compile kernel to use docker and USB Camera) with :

   ```dpkg -i linux-image-4.19.125-mtk_4-2_arm64.deb```

3. [Get started with the coral dev board mini](https://coral.ai/docs/dev-board-mini/get-started/)
   
   ! Before launch "sudo apt-get update" execute this line :
   
   ```sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys B53DC80D13EDEF05```

4. Install dependencies
   ```
   cd IAfinal
   bash requirement.sh
   ```

# How to use ?

To run cumulative counting with a Tensorflowlite person detection model use the [`detect.py` script](detect.py).

To run the program with the stream [`Streamdetect.py` script](Streamdetect.py).

## Arguments
   ```
    #You can change the name of the model here :
    default_model = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite' 
    default_labels = 'coco_labels.txt'

   --model -> 'model path' 
                    default=os.path.join(default_model)
   --labels -> 'label file path'
                        default=os.path.join(default_labels)
   --top_k' -> 'number of categories with highest score to display' type = int()
                        default=10
   --camera_idx -> 'Index of which video source to use'  type = int()
                        default = 0
   --threshold' -> ' classifier score threshold' type=float
                        default=0.1
   ```
## Sort Tracker

The tracking script name is ['sort.py' script](sort.py).

Explanation of the different arguments :
 ```
    #Create an instance :
        mot_tracker=Sort()

    #Update the tracker :
        trdata = mot_tracker.update(detections) #trdata content the tarcker's boundingbox

    #You can also create the tracker with different parameter :
    mot_tracker = Sort(max_age,min_hits,iou_threshold) 
   ```
    max_age: This is the maximum number of frames that an object can be considered as still being tracked even if it's not detected in some frames. After this number of frames, the object will be considered as lost.

    min_hits: This is the minimum number of detections required for an object to be tracked. Before reaching this number, the object is considered as not being tracked.

    iou_threshold: This is the intersection over union (IoU) threshold used to evaluate if two detection regions overlap enough to be considered as matching.

By using these parameters, the Sort tracker is able to track objects detected across frames of a video using matching and prediction algorithms.

## Motion sensor to reduce energy consumption

You can already use [HRSR 602](https://www.amazon.com/-/es/MH-SR602-movimiento-Piroel%C3%A9ctrico-Infrarrojos-Interruptor/dp/B07Z45RMZV) to decrease energy consumption :

![](Doc/HRSR602.jpg)

You can change the input GPIO by the following line :
 ```
        button = GPIO("/dev/gpiochip0", 13, "in")  # pin 36
```
Follow the recommandation on [coral.ai to connect pins](https://coral.ai/docs/dev-board-mini/gpio/)

## Send data by Alhora
You can change the input UART by the following line :
 ```
    port = serial.Serial("/dev/ttyS1", baudrate=9600, timeout =1)
```
or you can test with [simpleUartCoral script](simpleUartCoral.py)

License / Base on
----------------------
[Sort project](https://github.com/abewley/sort/tree/master)






