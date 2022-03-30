./gen-records.py --hug=$1 --sm=start --cf $*
./gen-records.py --hug=$1 --sm=end --cf $*
./gen-records.py --hug=$1 --sm=se --cf $*
./gen-records.py --hug=$1 --sm=mid --cf $*
./gen-records.py --hug=$1 --sm=start --stride=128 --cf $*
./gen-records.py --hug=$1 --sm=start --stride=0 --cf $*
./gen-records.py --hug=$1 --sm=start --stride=64 --cf $*
./gen-records.py --hug=$1 --sm=start --stride=256 --cf $*
./gen-records.py --hug=$1 --sm=start --stride=384 --cf $*
#./gen-records.py --hug=$1 --sm=se2 --cf $*
#./gen-records.py --hug=$1 --sm=se3 --cf $*
#./gen-records.py --hug=$1 --sm=se4 --cf $*
#./gen-records.py --hug=$1 --sm=se5 --cf $*
#./gen-records.py --hug=$1 --sm=se6 --cf $*
#./gen-records.py --hug=$1 --sm=se7 --cf $*

