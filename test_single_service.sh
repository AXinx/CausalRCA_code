for((i=4;i<=4;i++));  
do 
	for((j=1;j<=10;j++));
	do
	echo "$j"
	python train_single_service.py $i;
      	done;
echo "-----one service finish-----"	
done  
