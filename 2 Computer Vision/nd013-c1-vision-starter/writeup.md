# Data
## Download tfrecords
In first step I modified `download_process.py` to ignore files which were downloaded earlier.

## Exploratory Data Analysis and Split Creation
### Display 10 images
![ten_images](result/ten_frames.png)

### Create the splits
Create cross validation split from  download tfrecords. 
To save disk space I created soft link in [script]('create_splits.py') .

## Training
### Train on base parameters.
When I execute train scripts I had some problems. 
* Out of memory. To avoid this problem batch_size was decreased to 3. 
* TypeError in evaluation step. 
This problem was fixed when `metrics_set` was changed by `pascal_voc_detection_metrics`. 
[Link to knowledge forum.](https://knowledge.udacity.com/questions/657618) 

```
 PascalBoxes_Precision/mAP@0.5IOU: 0.046190
 PascalBoxes_PerformanceByCategory/AP@0.5IOU/vehicle: 0.060793
 PascalBoxes_PerformanceByCategory/AP@0.5IOU/pedestrian: 0.031586
 PascalBoxes_PerformanceByCategory/AP@0.5IOU/cyclist: nan
 Loss/localization_loss: 0.473013
 Loss/classification_loss: 0.378682
 Loss/regularization_loss: 0.518820
 Loss/total_loss: 1.370515
 ```
![ten_images](result/first_training.png)


Let use this result as started. 

`Note`. Cyclists not detected.


### Modifications.
#### Resnet101.
In first attempt I change `feature_extractor` from `resnet50` to `resnet101` in [pipeline](training/reference_resnet101/pipeline_new.config).
Unfortunately this method use more memory. 
`batch_size` was decreased to 2   

Final result was worse than with `resnet50`. mAP was smaller, losses is higher. Cyclist still not detected.

![resnet_101](result/training_resnet_101.png)

|Metric|reference|reference_resnet101|
|---|---|---|
|mAP@0.5IOU|0.046190|0.019115|
|vehicle|0.060793|0.038229|
|pedestrian|0.031586|0.000000|
|cyclist|nan|nan|
|localization_loss|0.473013|0.524127|
|classification_loss|0.378682|0.483823|
|regularization_loss|0.518820|0.606705|
|total_loss|1.370515|1.614655|

#### Chose augmentations
Let chose which augmentations are usefully.
On [github](https://github.com/tensorflow/models/blob/master/research/object_detection/configs/tf2/centernet_resnet50_v1_fpn_512x512_kpts_coco17_tpu-8.config)
I found 4 additional augmentations. Let compare them. 

After 5000 steps I get this results.

|Metric|base|brightness|contrast|hue|saturation|
|---|---|---|---|---|---|
|mAP@0.5IOU|0.014060|0.000938|0.053021|0.030180|0.000013|
|vehicle|0.028109|0.001875|0.078981|0.053272|0.000026|
|pedestrian|0.000010|0.000000|0.027062|0.007089|0.000000|
|cyclist|nan|nan|nan|nan|nan|
|localization_loss|0.590533|0.667572|0.427130|0.526328|1.027236|
|classification_loss|0.513630|0.898123|0.333566|0.363240|4.166665|
|regularization_loss|0.692396|1.782644|0.285555|0.472678|27997147136.000000|
|total_loss|1.796558|3.348339|1.046251|1.362247|27997147136.000000|

Pipelines with hue and contrast were found to be more effective. Pipelines with brightness and saturation were not.

#### Train with augmentations
I add to base [pipeline](training/reference_aug/pipeline_new.config) two augmentation:
* random_adjust_hue
* random_adjust_contrast

![resnet_101](result/training_with_aug.png)

|Metric|reference|reference_aug|
|---|---|---|
|mAP@0.5IOU|0.046190|0.053005|
|vehicle|0.060793|0.066437|
|pedestrian|0.031586|0.039574|
|cyclist|nan|nan|
|localization_loss|0.473013|0.466569|
|classification_loss|0.378682|0.346981|
|regularization_loss|0.518820|0.393930|
|total_loss|1.370515|1.207480|

### Known issues
* Small memory on gpu.
* Metrics_set


### Improvements suggestions.
* Compare with another feature_extractor.
* Use more powerful hardware. (Ten hours per full pipeline iteration is very slow).
* Use another augmentations.
* Use bigger dataset.