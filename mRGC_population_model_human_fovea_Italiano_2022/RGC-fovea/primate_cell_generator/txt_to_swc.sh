awk '{  printf $8+1 " "; \
        if ($1=="0") \
            printf "1"; \
        else if ($1=="1") \
            printf "3"; \
        else printf "2"; \
        print " " $2 " "$3 " " $4 " " $10 " " $9+1 }' \
$1 | sed 's/ 0$/ -1/' | sed 's/^2 3/2 1/' > $2 