for (( i=0; i<5; i++ ))
do
  ./main.py --allnew --fold=$i $*
  ./main.py --folds_metrics $* 
done
#./main.py --online $*
