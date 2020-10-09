python eval_fan_crf.py --data_eval_dir './data/300VW/category1/'\
 --eval_tfrecords_file 'thrVW_cat1_62846.tfrecords'\
 --model_dir './pretrained_models/300wlptrain/'\
 --model_name 'model_300wlptrain.ckpt'\
 --facemodel_path './facemodel/DM68_wild34.mat'\
 --eval_num 62846\
 --offset=-0.5\
 --save_result_name './results/300vw1.mat'