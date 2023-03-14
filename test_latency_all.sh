output11=cpu_latency_gamma5.txt;

echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output11;

for((i=0;i<=6;i++));
do
echo $i | tee -a $output11;
for((j=1;j<=10;j++));
do
	echo $j | tee -a $output11;
	python train_latency.py --indx=$i --atype='cpu-hog1_' --gamma=0.5 | tee -a $output11;
done;
echo "-----one service finish-----" | tee -a $output11;
done
echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output11;


output12=cpu_latency_gamma75.txt;

echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output12;

for((i=0;i<=6;i++));
do
echo $i | tee -a $output12;
for((j=1;j<=10;j++));
do
	echo $j | tee -a $output12;
	python train_latency.py --indx=$i --atype='cpu-hog1_' --gamma=0.75 | tee -a $output12;
done;
echo "-----one service finish-----" | tee -a $output12;
done
echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output12;


output13=cpu_latency_eta100.txt;

echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output13;

for((i=0;i<=6;i++));
do
echo $i | tee -a $output13;
for((j=1;j<=10;j++));
do
	echo $j | tee -a $output13;
	python train_latency.py --indx=$i --atype='cpu-hog1_' --eta=100 | tee -a $output13;
done;
echo "-----one service finish-----" | tee -a $output13;
done
echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output13;


output14=cpu_latency_eta1000.txt;

echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output14;

for((i=0;i<=6;i++));
do
echo $i | tee -a $output14;
for((j=1;j<=10;j++));
do
    echo $j | tee -a $output14;
    python train_latency.py --indx=$i --atype='cpu-hog1_' --eta=1000 | tee -a $output14;
done;
echo "-----one service finish-----" | tee -a $output14;
done
echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output14;


output1=memory_latency_gamma5.txt;

echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output1;

for((i=0;i<=6;i++));
do
echo $i | tee -a $output1;
for((j=1;j<=10;j++));
do
    echo $j | tee -a $output1;
    python train_latency.py --indx=$i --atype='memory-leak1_' --gamma=0.5 | tee -a $output1;
done;
echo "-----one service finish-----" | tee -a $output1;
done
echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output1;


output2=memory_latency_gamma75.txt;

echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output2;

for((i=0;i<=6;i++));
do
echo $i | tee -a $output2;
for((j=1;j<=10;j++));
do
    echo $j | tee -a $output2;
    python train_latency.py --indx=$i --atype='memory-leak1_' --gamma=0.75 | tee -a $output2;
done;
echo "-----one service finish-----" | tee -a $output2;
done
echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output2;


output3=memory_latency_eta100.txt;

echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output3;

for((i=0;i<=6;i++));
do
echo $i | tee -a $output3;
for((j=1;j<=10;j++));
do
    echo $j | tee -a $output3;
    python train_latency.py --indx=$i --atype='memory-leak1_' --eta=100 | tee -a $output3;
done;
echo "-----one service finish-----" | tee -a $output3;
done
echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output3;


output4=memory_latency_eta1000.txt;

echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output4;

for((i=0;i<=6;i++));
do
echo $i | tee -a $output4;
for((j=1;j<=10;j++));
do
    echo $j | tee -a $output4;
    python train_latency.py --indx=$i --atype='memory-leak1_' --eta=1000 | tee -a $output4;
done;
echo "-----one service finish-----" | tee -a $output4;
done
echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output4;


output5=network_latency_gamma5.txt;

echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output5;

for((i=0;i<=6;i++));
do
echo $i | tee -a $output5;
for((j=1;j<=10;j++));
do
    echo $j | tee -a $output5;
    python train_latency.py --indx=$i --atype='latency1_' --gamma=0.5 | tee -a $output5;
done;
echo "-----one service finish-----" | tee -a $output5;
done
echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output5;


output6=network_latency_gamma75.txt;

echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output6;

for((i=0;i<=6;i++));
do
echo $i | tee -a $output6;
for((j=1;j<=10;j++));
do
    echo $j | tee -a $output6;
    python train_latency.py --indx=$i --atype='latency1_' --gamma=0.75 | tee -a $output6;
done;
echo "-----one service finish-----" | tee -a $output6;
done
echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output6;


output7=network_latency_eta100.txt;

echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output7;

for((i=0;i<=6;i++));
do
echo $i | tee -a $output7;
for((j=1;j<=10;j++));
do
    echo $j | tee -a $output7;
    python train_latency.py --indx=$i --atype='latency1_' --eta=100 | tee -a $output7;
done;
echo "-----one service finish-----" | tee -a $output7;
done
echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output7;


output8=network_latency_eta1000.txt;

echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output8;

for((i=0;i<=6;i++));
do
echo $i | tee -a $output8;
for((j=1;j<=10;j++));
do
    echo $j | tee -a $output8;
    python train_latency.py --indx=$i --atype='latency1_' --eta=1000 | tee -a $output8;
done;
echo "-----one service finish-----" | tee -a $output8;
done
echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output8;
