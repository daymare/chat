
#python3 main.py

python3 main.py 2>&1 | tee out.txt; vim '+ normal G zR' out.txt


