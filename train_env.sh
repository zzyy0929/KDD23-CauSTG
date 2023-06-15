for env in 1 2 3 4;do
python train.py --gcn_bool --adjtype doubletransition --env $env --ifenv 1
done
python min_pool.py
python train.py --gcn_bool --adjtype doubletransition --env 5 --ifenv 0 