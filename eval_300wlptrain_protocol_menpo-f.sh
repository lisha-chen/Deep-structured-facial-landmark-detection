python eval_fan_crf.py --data_eval_dir './data/menpo/'\
 --eval_tfrecords_file 'menpo_train_frontal_6679.tfrecords'\
 --model_dir './pretrained_models/300wlptrain/'\
 --model_name 'model_300wlptrain.ckpt'\
 --facemodel_path './facemodel/DM68_wild34.mat'\
 --eval_num 6679\
 --save_result_name './results/menpo-f.mat'