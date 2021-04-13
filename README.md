Run in Anaconda:
```
conda create -n autism -c default matplotlib keras tensorflow keras-applications pydotplus
conda activate autism
pip install sklearn keras-vggface
python face.py -h
python face.py
python face.py -T -m best_hdf5_model_file_from_trainng
python face.py -O -m best_hdf5_model_file_from_trainng
python face.py -T -m best_hdf5_model_file_from_tuning
```