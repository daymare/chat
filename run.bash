

# ensure unique filename

number=0
base_name=outputs/out
fname=$base_name.txt

while [ -e "$fname" ]; do
    printf -v fname -- '%s-%02d.txt' "$base_name" "$(( ++number ))"
done


args=()
for var in "$@"; do
    # ignore the arguments specific to this bash script
    [ "$var" != '--notee' ] && [ "$var" != '--pdb' ] && args+=("$var")
done

if [[ "$*" == *--notee* ]]
then
    python3 main.py "${args}" 2>&1
elif [[ "$*" == *--pdb* ]]
then
    pdb3 main.py "${args}"
else
    python3 main.py $@ 2>&1 | tee $fname; vim '+ normal G zR' $fname
fi

