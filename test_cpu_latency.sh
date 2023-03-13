output1=cpu_latency_gamma5.txt;

echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output1;

for((i=0;i<=6;i++));
do
echo $i | tee -a $output1;
for((j=1;j<=10;j++));
do
	echo $j | tee -a $output1;
	python train_latency.py --indx=$i --atype='cpu-hog1_' --gamma=0.5 | tee -a $output1;
done;
echo "-----one service finish-----" | tee -a $output1;	
done
echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output1;



output2=cpu_latency_gamma75.txt;

echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output2;

for((i=0;i<=6;i++));
do
echo $i | tee -a $output2;
for((j=1;j<=10;j++));
do
	echo $j | tee -a $output2;
	python train_latency.py --indx=$i --atype='cpu-hog1_' --gamma=0.75 | tee -a $output2;
done;
echo "-----one service finish-----" | tee -a $output2;
done
echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output2;



output3=cpu_latency_eta100.txt;

echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output3;

for((i=0;i<=6;i++));
do
echo $i | tee -a $output3;
for((j=1;j<=10;j++));
do
	echo $j | tee -a $output3;
	python train_latency.py --indx=$i --atype='cpu-hog1_' --eta=100 | tee -a $output3;
done;
echo "-----one service finish-----" | tee -a $output3;
done
echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output3;


output4=cpu_latency_eta1000.txt;
echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output4;
for((i=0;i<=6;i++));
do
	echo $i | tee -a $output4;
	python train_latency.py --indx=$i --atype='cpu-hog1_' --eta=1000 | tee -a $output4;
done;
echo "-----one service finish-----" | tee -a $output4;
done
echo $(date +%Y-%m-%d" "%H:%M:%S) | tee -a $output4;
