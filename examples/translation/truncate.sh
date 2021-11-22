for filename in ./iwslt14.tokenized.de-en/tmp/train*; do
    [ -e "$filename" ] || continue
    # ... rest of the loop body
    a="${filename}_long"
    echo $a
    mv $filename $a
    head -2500 $a > $filename
done
