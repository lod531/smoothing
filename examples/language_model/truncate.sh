for filename in ./wikitext-103/*; do
    [ -e "$filename" ] || continue
    # ... rest of the loop body
    a="${filename}_long"
    echo $filename
    mv $filename $a
    head -2500 $a > $filename
    rm $a
done
