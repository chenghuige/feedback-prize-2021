./gen-records2.py --hug=longformer --aug=$1 --max_len=1280 --cf
./gen-records2.py --hug=longformer --aug=$1 --max_len=1536 --cf
./gen-records2.py --hug=longformer --aug=$1 --cf
./gen-records2.py --aug=$1 --hug=roberta --cf
./gen-records2.py --aug=$1 --hug=roberta --cf --sm=end
./gen-records2.py --aug=$1 --hug=roberta --cf --sm=se
./gen-records2.py --aug=$1 --hug=roberta --cf --sm=mid
./gen-records2.py --aug=$1 --hug=electra --cf
./gen-records2.py --aug=$1 --hug=electra --cf --sm=end
./gen-records2.py --aug=$1 --hug=electra --cf --sm=se
./gen-records2.py --aug=$1 --hug=electra --cf --sm=mid
./gen-records2.py --aug=$1 --hug=bart --cf
./gen-records2.py --aug=$1 --hug=bart --cf --sm=end
./gen-records2.py --aug=$1 --hug=bart --cf --sm=se
./gen-records2.py --aug=$1 --hug=bart --cf --sm=mid

  
