## 单机
直接运行 python tf_single_train.py
(GPU下运行方式一样, 使用gpu镜像会自动使用gpu进行训练)

## 集群
首先执行 python tf_parse_clusterspec.py获取集群的相关信息，包括ps_hosts,worker_hosts等
之后执行 python tf_dis_train.py --job_name=worker --task_index=0 --ps_hosts=xxx --worker_hosts=xxx
