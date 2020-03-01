# discrete_intonation
Code for paper titled
"[Perception of prosodic variation for speech synthesis using an unsupervised discrete representation of F0](https://arxiv.org/abs/2003.06686)"
accepted to the [International Conference on Speech Prosody 2020](https://sp2020.jpn.org).

Speech samples are available at
[zackhodari.github.io/SP_2020_discrete_intonation](http://zackhodari.github.io/SP_2020_discrete_intonation.html).


The models in this repo are written for the [`Morgana`](https://github.com/ZackHodari/morgana) speech synthesis toolkit.



# Tool setup
```bash
pip install git+https://github.com/zackhodari/tts_data_tools
pip install git+https://github.com/zackhodari/morgana
git clone https://github.com/ZackHodari/discrete_intonation.git

DATA_DIR=data/Blizzard2017

cd discrete_intonation
mkdir -p ${DATA_DIR}
mkdir experiments
```



# Data setup
The code provided assumes data is extracted with [`tts_data_tools`](https://github.com/ZackHodari/tts_data_tools), this
requires
- wavfiles at the desired frame-rate (16kHz in the paper)
- label files with time alignments (label_state_align)
- Festival Utterance structures so POS tags can be extracted in order to define phrases using the
[chinks_and_chunks](chinks_and_chunks_splitter.py) parser.

Note: Extracting chinks_and_chunks phrases requires part-of-speech tags and surface forms with the same tokenisation as
in the forced aligned phone sequence. If using Festival as your front-end, the simplest way to achieve this requires
access to the original voice: run dumpfeats specifying POS (and a couple other indexing-related) features for
extraction. The input `--lab_dir_with_pos` to the [chinks_and_chunks](chinks_and_chunks_splitter.py) parser must contain
one line per phone.

```bash
# Train split - extract features
tdt_process_dataset \
    --lab_dir ${DATA_DIR}/label_state_align \
    --wav_dir ${DATA_DIR}/wav_16000 \
    --out_dir ${DATA_DIR}/train \
    --question_file questions-unilex_dnn_600.hed \
    --id_list ${DATA_DIR}/train_file_id_list.scp \
    --state_level \
    --subphone_feat_type full \
    --calculate_normalisation \
    --normalisation_of_deltas

# Valid split - extract features
tdt_process_dataset \
    --lab_dir ${DATA_DIR}/label_state_align \
    --wav_dir ${DATA_DIR}/wav_16000 \
    --out_dir ${DATA_DIR}/valid \
    --question_file questions-unilex_dnn_600.hed \
    --id_list ${DATA_DIR}/valid_file_id_list.scp \
    --state_level \
    --subphone_feat_type full

# Extract POS tags and surface forms
tdt_utt_to_lab \
    --festival_dir ~/tools/festival \
    --utt_dir ${DATA_DIR}/utt \
    --id_list ${DATA_DIR}/all_file_id_list.scp \
    --out_dir ${DATA_DIR}/label_POS \
    --label_feats label_POS.feats \
    --label_full_awk label-POS.awk

# Train split - create chinks and chunks phrases
python chinks_and_chunks_splitter.py \
    --lab_dir_with_pos ${DATA_DIR}/label_POS \
    --wav_dir ${DATA_DIR}/wav_16000 \
    --lab_dir ${DATA_DIR}/label_state_align \
    --id_list ${DATA_DIR}/train_file_id_list.scp \
    --out_dir ${DATA_DIR}/train

# Valid split - create chinks and chunks phrases
python chinks_and_chunks_splitter.py \
    --lab_dir_with_pos ${DATA_DIR}/label_POS \
    --wav_dir ${DATA_DIR}/wav_16000 \
    --lab_dir ${DATA_DIR}/label_state_align \
    --id_list ${DATA_DIR}/valid_file_id_list.scp \
    --out_dir ${DATA_DIR}/valid
```



# Training models
```bash
python f0_phrase_AE.py --experiment_name phrase_AE \
    --data_root ${DATA_DIR} \
    --train_dir train --train_id_list train_file_id_list.scp \
    --valid_dir valid --valid_id_list valid_file_id_list.scp \
    --test \
    --test_dir f0_reaper_world --test_id_list id_lists/test_1phrase_id_list.scp \
    --end_epoch 100 \
    --num_data_threads 4 \
    --learning_rate 0.005 \
    --batch_size 32 \
    --lr_schedule_name noam \
    --lr_schedule_kwargs "{'warmup_steps':1800}" \
    --analysis_kwargs "{'n_clusters':20}"

python f0_phrase_VAMP.py --experiment_name phrase_VAMP \
    --data_root ${DATA_DIR} \
    --train_dir train --train_id_list train_file_id_list.scp \
    --valid_dir valid --valid_id_list valid_file_id_list.scp \
    --test \
    --test_dir valid --test_id_list valid_1phrase_id_list.scp \
    --end_epoch 100 \
    --num_data_threads 4 \
    --learning_rate 0.005 \
    --batch_size 32 \
    --lr_schedule_name noam \
    --lr_schedule_kwargs "{'warmup_steps':1800}" \
    --kld_wait_epochs 5 \
    --kld_warmup_epochs 20 \
    --model_kwargs "{'kld_weight':0.001,'pseudo_inputs_seq_lens':{'start':50,'stop':500,'step':50,'repeat':2}}" \
    --analysis_kwargs "{'modes':list(range(20))}"
```

