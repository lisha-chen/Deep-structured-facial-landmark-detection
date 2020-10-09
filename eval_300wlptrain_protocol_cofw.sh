python eval_fan_crf.py --data_eval_dir './data/COFW/test/'\
 --eval_tfrecords_file 'COFW_test_507.tfrecords'\
 --model_dir './pretrained_models/300wlptrain/'\
 --model_name 'model_300wlptrain.ckpt'\
 --facemodel_path './facemodel/DM68_wild34.mat'\
 --eval_num 507\
 --save_result_name './results/cofw.mat'