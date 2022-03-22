# ADL_HW1

下載 Glove embedding

```
wget http://nlp.stanford.edu/data/glove.840B.300d.zip -O glove.840B.300d.zip
unzip glove.840B.300d.zip
```

## Intent classification
1.  前處理
``` 
cd intent
python preprocess.py --data_dir <data_directory> --output_dir <output_directory> --glove_path <globe_path>
```

2.  開始訓練模型 
```
python train.py --device <device> ...
```

## Slot tagging
1.  前處理
```
cd slot
python preprocess.py --data_dir <data_directory> --output_dir <output_directory> --glove_path <globe_path>
```

2.  開始訓練模型
```
python train.py --device <device> ...
```
