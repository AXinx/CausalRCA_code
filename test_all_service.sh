for((i=0;i<=6;i++));  
do 
	for((j=1;j<=10;j++));
	do
	echo "$j"
	python train_all_services.py $i | tee -a all_service_res.txt;
      	done;
echo "-----one service finish-----"	
done  
