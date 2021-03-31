本项目使用Pytorch实现（我认为的）目标检测中的里程碑模型。
<br/>在自身学习之余，我希望能帮助读者们通过阅读风格一致的代码，来学习、使用这些模型，减少学习成本。</br>
因为能力有限或者其他原因，我不能保证模型都完美地复现论文的细节，只希望能抓（一些）大放小。同时，我本人也不期望这里的模型能达到生产级别。
同时我想提醒读者，本项目仅仅提供了一些模型上的信息，事实上本项目涉及的这些里程碑模型在别的地方也很有可取之处，甚至起到了至关重要的作用。比如yolo v2在ImageNet的finetune过程，这些
都是本项目力所不逮的地方。

This Repository uses Pytorch to implement milestone models(in my opinion) of objection detection.
<br/>I hope it can help you study and use these great models with less learning cost by same style codes.</br>
Due to limited capacity, I won't implement all the details of origin papers but only focus on (some)key points. 
Besides, models here are not necessarily good enough to be used directly for production environment.  And I really want to remind you that this repository only provide some infomation of models, while these related milestone actually did something great beyond model architecture, which may be even more important than models. For example, yolo v2 finetune model on ImageNet to make model adaptive to 448 * 448 image. 


# progress
+ [ ] 01_RCNN
+ [ ] 02_OverFeat
+ [ ] 03_Fast_RCNN
+ [x] 04_Faster_RCNN
+ [ ] 05_OHEM
+ [x] 06_YOLO_v1
+ [ ] 07_SSD
+ [ ] 08_R_FCN
+ [ ] 09_YOLO_v2
+ [ ] 10_FPN
+ [ ] 11_RetinaNet
+ [ ] 12_Mask_RCNN
+ [ ] 13_YOLO_v3
+ [ ] 14_RefineDet
+ [ ] 15_M2Det

# requirements
python 3.7
<br/>opencv 3.4.2</br>
pytorch 1.4.0

# running
Basically you can run train_test.py in every folder.

# note
I have some unnecessary preference on format and order and something like that. I hope the folders' name wouldn't
bother you. 

# reference
