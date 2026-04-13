# STF: Steady and Transient Factorization for Sparse Time-Aware QoS Prediction

## 1.Experimental Details

### 1.1  Experimental Setting

We conduct experiments under different training data densities, including 1%, 2%, 4%, 6%, and 8%, to evaluate the STF model and all baseline methods. Both the STF model and all baseline methods are trained under two learning rate configurations, with initial learning rates set to 0.0001 and 0.001, respectively. When the initial learning rate is set to 0.001, the learning rate is decayed by a factor of 0.5 every 10 epochs. The maximum number of training epochs for all methods is set to 100, and the final 10 epochs are used for testing.For all models under different training data densities, the final results are selected as the best testing performance achieved across the two learning rate settings.

Model performance is evaluated using MAE and RMSE metrics. All experiments are conducted on a system equipped with an Intel Xeon Silver 4210R CPU (2.40 GHz), a Tesla V100 GPU, 128 GB of RAM, and running Ubuntu 24.04.

### 1.2  Dataset Description

We use the WS-DREAM Dataset2 as the benchmark dataset in our experiments. This dataset contains large-scale, real-world Quality of Service (QoS) evaluation records collected from 142 geographically distributed users interacting with 4,500 Web services across 64 consecutive time slices, with each time slice recorded at 15-minute intervals. The fine-grained temporal granularity enables the analysis of dynamic QoS variations over time, while the large-scale user–service interaction space results in a highly sparse observation matrix, making it well-suited for evaluating QoS prediction models under realistic conditions.

The dataset can be accessed and downloaded from the official repository:
http://wsdream.github.io/dataset/wsdream_dataset2.html

After downloading, please place the files `rtdata.txt` (responsetime) and `tpdata.txt` (throughput) into the directory `dataset\ws-time\wsdream` to ensure compatibility with the experimental framework and to support reproducible evaluation across different QoS prediction models.

Install environment dependencies using the following command:
```shell
pip install -r requirements.txt
```

### 1.3 Data Generation for Model Training

Following established practice, we implement random training and test set splits within each time slice using pre-defined density ratios, preventing temporal leakage. For example, the density = 1% indicates that only 1% of observed entries per slice are used for training while the remainder are held out for testing. The objective of QoS prediction is to assess the model’s capability to accurately interpolate missing entries within partially observed slices, as opposed to extrapolating to future slices. In this setting, the model is first trained on the training entries and then fixed, after which it is directly applied to predict the held-out test entries.


### 1.4 Training STF Model

Run the following commands to train the STF
```shell
# STF on responsetime dataset
python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name stf --loss_type mae --embed_dim 16 --cuda 0
# STF on throughput dataset
python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name stf --loss_type mae --embed_dim 32 --cuda 0
```

### 1.5 Baseline Reproduction

Use the following commands to reproduce baseline models:
```shell
#wspred on responsetime dataset
python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.1 --model_name wspred --loss_type mae --embed_dim 16 --cuda 0
#wspred on throughput dataset
python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.1 --model_name wspred --loss_type mae --embed_dim 16 --cuda 0

#costco on responsetime dataset
python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name costco --loss_type mae --embed_dim 16 --cuda 0
#costco on throughput dataset
python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name costco --loss_type mae --embed_dim 16 --cuda 0

#deeptsqp on responsetime dataset
python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name deeptsqp --loss_type mae --embed_dim 16 --cuda 0
#deeptsqp on throughput dataset
python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name deeptsqp --loss_type mae --embed_dim 16 --cuda 0

#gm on responsetime dataset
python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name gm --loss_type mae --embed_dim 16 --cuda 0
#gm on throughput dataset
python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name gm --loss_type mae --embed_dim 16 --cuda 0

#ntf on responsetime dataset
python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name ntf --loss_type mae --embed_dim 16 --cuda 0
#ntf on throughput dataset
python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name ntf --loss_type mae --embed_dim 16 --cuda 0

#scatsf on responsetime dataset
python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name scatsf --loss_type mae --embed_dim 16 --cuda 0
#scatsf on throughput dataset
python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name scatsf --loss_type mae --embed_dim 16 --cuda 0

#plmf on responsetime dataset
python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name plmf --loss_type mae --embed_dim 16 --cuda 0
#plmf on throughput dataset
python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name plmf --loss_type mae --embed_dim 16 --cuda 0

#rncf on responsetime dataset
python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name rncf --loss_type mae --embed_dim 16 --cuda 0
#rncf on throughput dataset
python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name rncf --loss_type mae --embed_dim 16 --cuda 0

#trcf  on responsetime dataset
python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name trcf --loss_type mae --embed_dim 16 --cuda 0
#trcf  on throughput dataset
python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name trcf --loss_type mae --embed_dim 16 --cuda 0

#tuipcc on responsetime dataset
python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name tuipcc --loss_type mae --embed_dim 16 --cuda 0
#tuipcc on throughput dataset
python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name tuipcc --loss_type mae --embed_dim 16 --cuda 0
```

# 2 Baseline

| Model    | Reference                                                                                                                                                                                                               |
|----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| WSPred   | [Y. Zhang, et al. WSPred: A Time-Aware Personalized QoS Prediction Framework for Web Services. ISSRE 2011.](https://ieeexplore.ieee.org/document/6132969)                                                |
| CoSTCo   | [H. Liu, et al. CoSTCo: A Neural Tensor Completion Model for Sparse Tensors, KDD, 2019.](https://liyaguang.github.io/papers/kdd19_CoSTCo.pdf)                                                    |
| DeepTSQP | [G. Zou, et aj. DeepTSQP: Temporal-aware service QoS prediction via deep neural network and feature integration, Knowledege-based Systems, 2022.](https://www.sciencedirect.com/science/article/pii/S0950705121011448) |
| NTF      | [X. Wu, et al: Neural Tensor Factorization for Temporal Interaction Learning. WSDM 2019.](https://dl.acm.org/doi/10.1145/3289600.3290998)                                                                              |
| PLMF     | [R. Xiong, et al: Personalized LSTM Based Matrix Factorization for Online QoS Prediction. ICWS 2017.](https://ieeexplore.ieee.org/document/8456329)                                                                    |
| RNCF     | [T. Liang, et al: Recurrent Neural Network Based Collaborative Filtering for QoS Prediction in IoV. TITS, 2022.](https://ieeexplore.ieee.org/document/9511331)                                                         |
| SCATSF   | [J. Zhou , et al. Spatial Context-Aware Time-Series Forecasting for QoS Prediction, TNSM, 2023.](https://ieeexplore.ieee.org/document/10057199/)                                                                       |
| GM       | [H. Wu et al. Effective Graph Modeling and Contrastive Learning for Time-Aware QoS Prediction, TSC, 2024.](https://ieeexplore.ieee.org/document/10713972)                                                            
| TUIPCC   | [E. Tong, et al. A Missing QoS Prediction Approach via Time-Aware Collaborative Filtering, TSC, 2022.](https://ieeexplore.ieee.org/document/9511220)                                                                   |
| TRCF     | [Z. Zheng, et al. TRCF: Temporal Reinforced Collaborative Filtering for Time-Aware QoS Prediction, TSC, 2024](https://ieeexplore.ieee.org/document/10314775)  
                                                                           |