df=/root/hdd-data/sift1b
qf=/root/hdd-data/sift1b
f=/root/hdd-data/sift1b/For_NeOPFA_new
gf=/root/hdd-data/sift1b

data=sift1b
n=1000000000
d=128
p=4096
nps=300
npage=300
r=100
K1=64
c=1
M=25
m=22
v=10
vv=38
net_i=5
nq=100
rr=1
flag=1
useOri=0

cd ../../cmake-build-release

make

pf=/root/hdd-data/sift1b/Index_Learning_code/models_${data}_d${M}_v${v}_${vv}
tf=/root/hdd-data/sift1b/pro_kmeans_data
kf=/root/hdd-data/sift1b/cluster/models_03
sf=/root/hdd-data/sift1b/step02/method_T6

for k in 10 20 30 40 50 60 70 80 90 100
do
sT=$((k * 3 / 2))
T=$((k * 6))
sudo sync
sudo echo 3 > /proc/sys/vm/drop_caches
./NeOPFA_PLUS_search_step01 ${data} ${n} ${nq} ${d} ${M} ${m} ${k} ${T} ${p} ${nps} ${r} \
${df}/data.ds \
${pf}/${data}_query_d${M}_v${v}_vv${vv}_${net_i}.txt \
${qf}/query.ds \
${f}/index_${M}_v${v}_vv${vv}_${net_i}_kmeans \
${gf}/groundtruth_100.txt \
${gf}/groundtruth_dist_100.txt \
${f}/result_step01_${M}_v${v}_vv${vv}_${net_i}.txt \
${rr} ${K1} ${c} ${flag} \
${tf}/${data}_new_base_d${M}_v${v}_vv${vv}_${net_i}.bin \
${tf}/ori_map_new.ds \
${kf}/cluster_k100_v755.txt \
${sf}/${data}_id_k${k}.txt \
${sf}/${data}_k${k}.txt \
${useOri} \
${sT} \
${npage}
done
