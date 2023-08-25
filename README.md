C:\Users\msc\Desktop\stocks-prediction-Machine-learning-RealTime-TensorFlow\venv\Scripts\python.exe C:\Users\msc\Desktop\stocks-prediction-Machine-learning-RealTime-TensorFlow\Tutorial\RUN_buy_sell_Tutorial_3W_5min_RT.py 
cuda_malloc_async

[1] Read data ohlcv data  path  ../d_price/600519_SH_1min.csv  Shape_raw:  (167986, 5)

[2] Calculating Technical indicators. stock:  600519_SH
[2.1] Calculated Technical indicators. stock:  600519_SH  Tech indicator count:  (167432, 316)

[3] Calculate the target_Y (ground true) ,what is the target to detect?  the highest and lowest peaks stock:  600519_SH
get_HT_pp df.shape: (167432, 294)  len_right:  72  len_left:  36
[3.1]target_Y.shape:  (167432, 3)
[3.2] DEBUG All technical indicators Extracted Count:  294

[4] Calculation of correlation strength. What are the +-100 best technical indicators and which are noise 
[4.1]uncorrelate_selection path:  data/columns_select/600519_SH__1min__test_A1__columns.pkl
	created_json_feature_selection path:  data/columns_select/600519_SH__1min__test_A1__corr.json
data/columns_select/600519_SH__1min__test_A1__corr.json
[4.2] Select the best technical patterns to train with features_W3 Extracted Count:  88  Names :  cycl_SINE_lead,mtum_PLUS_DM,olap_MIDPRICE,vola_TRANGE,vola_ATR,ichi_senkou_b,high,perf_ha,mtum_STOCH_Fa_kd,ma_WMA_10,volu_PVOL,cama_r3,mtum_QQEs_14_5_4236,volume,mtum_STCmacd_10_12_26_05,sti_LINEARREG_ANGLE,mtum_FISHERT_9_1,tend_PSARl_002_02,mtum_MINUS_DM,olap_BBAND_dif,sti_CORREL,vola_KCUe_20_2,clas_s3,mtum_MFI,vola_NATR,tend_PSARs_002_02,tend_renko_TR,olap_PWMA_10,ma_SMA_50,ichi_senkou_a,demark_s1,mtum_ADX,ti_mass_index_9_25,cama_s2,mtum_TRIX,mtum_QQEl_14_5_4236,ma_SMA_20,ti_konk_bl,sti_STDDEV,volu_NVI_1,olap_SAREXT,open,mtum_ER_10,cycl_PHASOR_quad,mtum_ADXR,ma_SMA_100,mtum_CFO_9,sti_VAR,vola_THERMOl_20_2_05,olap_BBAND_LOWER,ti_choppiness_14,vola_THERMO_20_2_05,olap_ALMA_10_60_085,mtum_PVOs_12_26_9,demark_pp,mtum_INERTIA_20_14,ti_kelt_20_lower,low,vola_THERMOma_20_2_05,vola_HWU,mtum_CTI_12,mtum_MACD,mtum_BOP,mtum_td_seq,cycl_PHASOR_inph,cama_s3,vola_KCLe_20_2,fibo_s1,volu_PVI_1,ma_TEMA_5,tend_PSARr_002_02,mtum_PVO_12_26_9,mtum_PVOh_12_26_9,mtum_DX,volu_PVT,day_minute,cycl_DCPHASE,close,tend_VHF_28,sti_BETA,mtum_MACD_fix,ichi_tenkan_sen,mtum_STOCH_kd,cycl_SINE_sine,cycl_DCPERIOD,olap_SINWMA_14,mtum_CG_10,sti_LINEARREG_INTERCEPT
[4.3] features_W3 index Dates: from  2020-09-01 11:26:00+00:00  to  2023-08-01 14:56:00+00:00  Shape:  (167432, 88)

[5] For correct training, for correct training the values must be normalised to between 0 and 1  
[5.1] Normalise_data path:  data/scalers/600519_SH__1min__test_A1__scalex.pkl

[6] Currently you have for each target_Y value a row of technical indicators, you add a 'window' to make the decision to predict whether the +-48 rows above will be taken (about 4 hours of previous indicators, splited in 5min).
[6.1] get_window_data Shapes Y: (167403, 3)  X:  (167403, 30, 88)

[7] data split between training and validation 
[7.1] Shapes: X_train:  (133922, 30, 88)  y_train:   (133922, 3)  index_train:   (133922,)

[8] Ground True data are unbalanced  Given that there is a lot of 'do nothing' 0, and very little 'do buy' 1 or 'do sell' 2, weight balancing is required, to give more importance to the minorities.  
[8.1] Class weight  path:  data/class_weight/600519_SH__1min__test_A1__class_weight.pkl  Dict:  {0: 0.34234690225671544, 1: 25.220715630885124, 2: 25.421791951404707}

[9] Creation of the TF model architecture. must respect the input_shape and output_shape and the 'softmax' , from there EXPERIMENT combinations 
[9.1] arrays to use the TF model  , input_shape:  (30, 88)  output_shape:  3
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 30, 144)           12816     
                                                                 
 dense_1 (Dense)             (None, 30, 72)            10440     
                                                                 
 dropout (Dropout)           (None, 30, 72)            0         
                                                                 
 dense_2 (Dense)             (None, 30, 36)            2628      
                                                                 
 dropout_1 (Dropout)         (None, 30, 36)            0         
                                                                 
 dense_3 (Dense)             (None, 30, 12)            444       
                                                                 
 flatten (Flatten)           (None, 360)               0         
                                                                 
 dense_4 (Dense)             (None, 3)                 1083      
                                                                 
=================================================================
Total params: 27411 (107.07 KB)
Trainable params: 27411 (107.07 KB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
None
[9.2] optimizer config :  {'name': 'Adam', 'weight_decay': None, 'clipnorm': None, 'global_clipnorm': None, 'clipvalue': None, 'use_ema': False, 'ema_momentum': 0.99, 'ema_overwrite_frequency': None, 'jit_compile': False, 'is_legacy_optimizer': False, 'learning_rate': 0.001, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-07, 'amsgrad': False}
You must install pydot (`pip install pydot`) and install graphviz (see instructions at https://graphviz.gitlab.io/download/) for plot_model to work.
[9.3] print diagram of the TF model Path:  outputs/plots/600519_SH__1min__test_A1__buysell.png

[10] Start training TF model  outputs/600519_SH__1min__test_A1__buysell.h5 
Epoch 1/90
5608/5608 [==============================] - 23s 4ms/step - loss: 0.7105 - categorical_accuracy: 0.6187 - val_loss: 0.8174 - val_categorical_accuracy: 0.6993
Epoch 2/90
5608/5608 [==============================] - 23s 4ms/step - loss: 0.4403 - categorical_accuracy: 0.8260 - val_loss: 0.7015 - val_categorical_accuracy: 0.7648
Epoch 3/90
5608/5608 [==============================] - 23s 4ms/step - loss: 0.3975 - categorical_accuracy: 0.8462 - val_loss: 0.3768 - val_categorical_accuracy: 0.8997
Epoch 4/90
5608/5608 [==============================] - 23s 4ms/step - loss: 0.3607 - categorical_accuracy: 0.8458 - val_loss: 0.6644 - val_categorical_accuracy: 0.7786
Epoch 5/90
5608/5608 [==============================] - 23s 4ms/step - loss: 0.3324 - categorical_accuracy: 0.8555 - val_loss: 0.3919 - val_categorical_accuracy: 0.8789
Epoch 6/90
5608/5608 [==============================] - 24s 4ms/step - loss: 0.3179 - categorical_accuracy: 0.8583 - val_loss: 0.3339 - val_categorical_accuracy: 0.9168
Epoch 7/90
5608/5608 [==============================] - 25s 4ms/step - loss: 0.2960 - categorical_accuracy: 0.8674 - val_loss: 0.2159 - val_categorical_accuracy: 0.9373
Epoch 8/90
5608/5608 [==============================] - 24s 4ms/step - loss: 0.2868 - categorical_accuracy: 0.8704 - val_loss: 0.3691 - val_categorical_accuracy: 0.8975
Epoch 9/90
5608/5608 [==============================] - 23s 4ms/step - loss: 0.2717 - categorical_accuracy: 0.8726 - val_loss: 0.4140 - val_categorical_accuracy: 0.8762
Epoch 10/90
5608/5608 [==============================] - 24s 4ms/step - loss: 0.2711 - categorical_accuracy: 0.8769 - val_loss: 0.2827 - val_categorical_accuracy: 0.9137
Epoch 11/90
5608/5608 [==============================] - 24s 4ms/step - loss: 0.2704 - categorical_accuracy: 0.8769 - val_loss: 0.3025 - val_categorical_accuracy: 0.9167
Epoch 12/90
5608/5608 [==============================] - 23s 4ms/step - loss: 0.2559 - categorical_accuracy: 0.8782 - val_loss: 0.2621 - val_categorical_accuracy: 0.9224
Epoch 13/90
5608/5608 [==============================] - 24s 4ms/step - loss: 0.2684 - categorical_accuracy: 0.8803 - val_loss: 0.2114 - val_categorical_accuracy: 0.9410
Epoch 14/90
5608/5608 [==============================] - 24s 4ms/step - loss: 0.2528 - categorical_accuracy: 0.8826 - val_loss: 0.2939 - val_categorical_accuracy: 0.9202
Epoch 15/90
5608/5608 [==============================] - 24s 4ms/step - loss: 0.2499 - categorical_accuracy: 0.8859 - val_loss: 0.2885 - val_categorical_accuracy: 0.9149
Epoch 16/90
5608/5608 [==============================] - 23s 4ms/step - loss: 0.2349 - categorical_accuracy: 0.8865 - val_loss: 0.2583 - val_categorical_accuracy: 0.9228
Epoch 17/90
5608/5608 [==============================] - 23s 4ms/step - loss: 0.2364 - categorical_accuracy: 0.8854 - val_loss: 0.2240 - val_categorical_accuracy: 0.9278
Epoch 18/90
5608/5608 [==============================] - 23s 4ms/step - loss: 0.2405 - categorical_accuracy: 0.8872 - val_loss: 0.2333 - val_categorical_accuracy: 0.9321
Epoch 19/90
5608/5608 [==============================] - 24s 4ms/step - loss: 0.2327 - categorical_accuracy: 0.8916 - val_loss: 0.3073 - val_categorical_accuracy: 0.9091
Epoch 20/90
5608/5608 [==============================] - 24s 4ms/step - loss: 0.2366 - categorical_accuracy: 0.8916 - val_loss: 0.3323 - val_categorical_accuracy: 0.9120
Epoch 21/90
5608/5608 [==============================] - 24s 4ms/step - loss: 0.2331 - categorical_accuracy: 0.8923 - val_loss: 0.2864 - val_categorical_accuracy: 0.9134
Epoch 22/90
5608/5608 [==============================] - 24s 4ms/step - loss: 0.2351 - categorical_accuracy: 0.8907 - val_loss: 0.2447 - val_categorical_accuracy: 0.9264
Epoch 23/90
5608/5608 [==============================] - 23s 4ms/step - loss: 0.2238 - categorical_accuracy: 0.8910 - val_loss: 0.3388 - val_categorical_accuracy: 0.8978
Epoch 23: early stopping
[10.1] Model initial_weights saved :  data/initial_weights/600519_SH__1min__test_A1__initial_weights.tf

[11] Do a predict and eval. Load path:  outputs/600519_SH__1min__test_A1__buysell.h5
[11.1] array to predict x_test.shape:  (33481, 30, 88)
5581/5581 [==============================] - 5s 958us/step
[11.2] read the evaluation:  (33481, 30, 88)
(33481,) (33481,)
              precision    recall  f1-score   support

           0       1.00      0.95      0.97     32573
           1       0.35      0.81      0.48       452
           2       0.30      0.89      0.44       456

    accuracy                           0.95     33481
   macro avg       0.55      0.88      0.63     33481
weighted avg       0.98      0.95      0.96     33481

outputs/model_info/600519_SH__1min__test_A1_.csv
               count   per%
y_pred y_test              
0      0       30924  92.36
2      0         967   2.89
1      0         682   2.04
2      2         406   1.21
1      1         364   1.09
0      1          88   0.26
       2          46   0.14
1      2           4   0.01
outputs/model_info/600519_SH__1min__test_A1__.info.txt


END

进程已结束，退出代码为 0
