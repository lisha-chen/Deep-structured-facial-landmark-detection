python eval_fan_crf.py --data_eval_dir './data/300w_train_val/val/'\
 --eval_tfrecords_file 'thrWtrain_val_689.tfrecords'\
 --model_dir './pretrained_models/300wtrain/'\
 --model_name 'model_300wtrain.ckpt'\
 --facemodel_path './facemodel/DM68_wild34.mat'\
 --eval_num 689\
 --save_result_name './results/300wtrain_val689.mat'