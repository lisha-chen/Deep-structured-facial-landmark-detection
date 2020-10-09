python eval_fan_crf.py --data_eval_dir './data/300VW/category2/'\
 --eval_tfrecords_file 'thrVW_cat2_32872.tfrecords'\
 --model_dir './pretrained_models/300wlptrain/'\
 --model_name 'model_300wlptrain.ckpt'\
 --facemodel_path './facemodel/DM68_wild34.mat'\
 --eval_num 32872\
 --offset=-0.5\
 --save_result_name './results/300vw2.mat'