if=/root/hdd-data/deep1b/For_NeOPFA_new
df=/root/hdd-data/deep1b
kf=/root/hdd-data/deep1b/forest_64_256_8
M=25
v=10
vv=38
net_i=9

data=deep1b
n=1000000000
d=96
p=4096
K1=64

pf=/root/hdd-data/deep1b/Index_Learning_code/models_${data}_d${M}_v${v}_${vv}

cd ../../cmake-build-release

make

./NeOPFA_PLUS_index ${data} ${n} ${d} ${M} ${p} \
${if}/index_${M}_v${v}_vv${vv}_${net_i}_kmeans \
${df}/data.ds \
${pf}/${data}_base_d${M}_v${v}_vv${vv}_${net_i}.bin \
${K1} \
${kf}/codes1.txt
