python eval_fan_crf.py --data_eval_dir './data/300VW/category3/'\
 --eval_tfrecords_file 'thrVW_cat3_27687.tfrecords'\
 --model_dir './pretrained_models/300wlptrain/'\
 --model_name 'model_300wlptrain.ckpt'\
 --facemodel_path './facemodel/DM68_wild34.mat'\
 --eval_num 27687\
 --offset=-0.5\
 --save_result_name './results/300vw3.mat'