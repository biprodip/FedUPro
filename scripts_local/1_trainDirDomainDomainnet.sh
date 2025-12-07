#CUDA_VISIBLE_DEVICES=0 python prompt-ours.py

round=10
local_epochs=1
num_clients=2 #per domain
choose="random"

domainnet_num_classes=10 #20
domainnet_batch_size=64
domainnet_learning_rate=0.01

gctx=16 # key context vectors

domainnet_datapath="../../FedPCL/data/domainnet"
cd ~/FedAPT/supervised_Asyn/

# domainnet_datapath="/datasets/work/d61-csirorobotics/work/pal194/data/domainnet/"
# cd "/scratch3/pal194/FedAPT/supervised_Asyn/"

non_iid='FNLN' #FNLI
alpha=0.1 #dirichlet param
seed=1
backbone='ViT-B/32'
# participation_rate=0.25 


# Fedupro__MMD_local_Experts_bayes____________________________________________________________________
algorithm="promptfl_MMD_1_local_Bayes_2_phase_loaded_${num_clients}_cl"
data="domainnet"
# log_file="logDir_${choose}_${data}/ablation_${algorithm}_${data}_${alpha}_${domainnet_num_classes}_${non_iid}_seed_${seed}.log"
log_file="logDir_${choose}_${data}/calibration_Fedupro_${data}_${alpha}_seed_${seed}.log"
echo "Output is being saved in ${log_file}"

python run_fedupro.py \
--alpha "$alpha" \
--batch_size "$domainnet_batch_size" \
--choose "$choose" \
--data "$data" \
--datapath "$domainnet_datapath" \
--learning_rate "$domainnet_learning_rate" \
--local_epochs "$local_epochs" \
--num_classes "$domainnet_num_classes" \
--num_clients "$num_clients" \
--gctx "$gctx" \
--non_iid "$non_iid" \
--seed "$seed" \
--backbone "$backbone" \
--mix_n_experts 7 \
--val_split 0.3 \
--save_pool domainnet_pool_VIT32_seed_1_alpha_0.1_10_epoch_.pkl.gz \
--round "$round" > "$log_file"
