### Get Traffic Light Dataset
  
You can get the traffic_lights dataset folder on auton cluster at:  
```
/zfsauton/datasets/ArgoRL/zheh/datasets/traffic_lights
```  
  
By default, put it as:  
```
AdelaiDet
└── datasets 
    └── traffic_lights
```  
  
There are three sets of annotations files in ''traffic_lights/annotations folder``, which are:  
1. instances_[test|train].json, 2-class detection (class 1: Green, 2: Red)  
2. instances4_[test|train].json, 4-class detection (class 1: Green, 2: Red, 3: Sign, 4: Car)  
3. instances5_[test|train].json, 5-class detection (class 1: Green, 2: Red, 3: Pole, 4: Sign, 5: Car)   

**Please make sure to adjust the config w.r.t. the dataset you use accordingly (e.g. NUM_CLASSES).**  


