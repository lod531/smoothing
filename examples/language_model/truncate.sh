for filename in ./wikitext-2/*; do
    [ -e "$filename" ] || continue
    # ... rest of the loop body
    a="${filename}_long"
    echo $filename
    mv $filename $a
    head -10000 $a > $filename
    rm $a
done
