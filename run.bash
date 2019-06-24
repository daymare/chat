

# ensure unique filename

number=0
base_name=outputs/out
fname=$base_name.txt

while [ -e "$fname" ]; do
    printf -v fname -- '%s-%02d.txt' "$base_name" "$(( ++number ))"
done

python3 main.py 2>&1 | tee $fname; vim '+ normal G zR' $fname

