python eval_fan_crf.py --data_eval_dir './data/300w_test/'\
 --eval_tfrecords_file 'thrWtest_600.tfrecords'\
 --model_dir './pretrained_models/300wlptrain/'\
 --model_name 'model_300wlptrain.ckpt'\
 --facemodel_path './facemodel/DM68_wild34.mat'\
 --eval_num 600\
 --save_result_name './results/300wtest.mat'