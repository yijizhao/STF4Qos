#! /bin/bash
default_seed=3305

#ResponseTime
python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.1 --model_name wspred --loss_type mae --embed_dim 16 --cuda 0

python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name costco --loss_type mae --embed_dim 16 --cuda 0

python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name deeptsqp --loss_type mae --embed_dim 16 --cuda 0

python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name gm --loss_type mae --embed_dim 16 --cuda 0

python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name ntf --loss_type mae --embed_dim 16 --cuda 0

python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name scatsf --loss_type mae --embed_dim 16 --cuda 0

python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name plmf --loss_type mae --embed_dim 16 --cuda 0

python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name rncf --loss_type mae --embed_dim 16 --cuda 0

python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name trcf --loss_type mae --embed_dim 16 --cuda 0

python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name tuipcc --loss_type mae --embed_dim 16 --cuda 0

python main.py --dataset_path dataset/ws-time/wsdream/rtdata.txt --data_task RT --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name stf --loss_type mae --embed_dim 16 --cuda 0

#Throughput
python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.1 --model_name wspred --loss_type mae --embed_dim 16 --cuda 0

python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name costco --loss_type mae --embed_dim 16 --cuda 0

python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name deeptsqp --loss_type mae --embed_dim 16 --cuda 0

python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name gm --loss_type mae --embed_dim 16 --cuda 0

python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name ntf --loss_type mae --embed_dim 16 --cuda 0

python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name scatsf --loss_type mae --embed_dim 16 --cuda 0

python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name plmf --loss_type mae --embed_dim 16 --cuda 0

python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name rncf --loss_type mae --embed_dim 16 --cuda 0

python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name trcf --loss_type mae --embed_dim 16 --cuda 0

python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name tuipcc --loss_type mae --embed_dim 16 --cuda 0

python main.py --dataset_path dataset/ws-time/wsdream/tpdata.txt --data_task TP --batch_size 1024 --epoch 100 --learn_rate 0.001 --model_name stf --loss_type mae --embed_dim 32 --cuda 0