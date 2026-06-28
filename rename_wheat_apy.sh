#!/bin/bash

DIR="/Users/sthitpragye/Desktop/Finance/SURGE/Wheat_APY/Wheat_Yield_District_wise_1998-2025"

cd "$DIR" || exit 1

for file in *.csv; do
    # Read cropyear from the first data row (2nd line), first column named cropyear
    cropyear=$(awk -F',' 'NR==2 {print $4}' "$file" | tr -d '"')

    # Extract the ending year (e.g. 99 from 1998-99)
    end=${cropyear#*-}

    # Convert to 4-digit year
    if [ "$end" -lt 50 ]; then
        newyear=$((2000 + 10#$end))
    else
        newyear=$((1900 + 10#$end))
    fi

    newname="${newyear}.csv"

    echo "$file  -->  $newname"
    mv "$file" "$newname"
done

echo "Done."


# #!/bin/bash
# set -e

# BASE="/Users/sthitpragye/Desktop/Finance/SURGE/Wheat_APY"

# echo "=== Renaming Area_Productivity_map-N.csv -> <1998+N>.csv (Area + Yield) ==="
# for dir in "Wheat_Area_District_wise_1998-2025" "Wheat_Yield_District_wise_1998-2025"; do
#     echo "--- $dir ---"
#     cd "$BASE/$dir"
#     for n in {1..27}; do
#         src="Area_Productivity_map-${n}.csv"
#         year=$((1998 + n))
#         dst="${year}.csv"
#         if [ -f "$src" ]; then
#             mv -v "$src" "$dst"
#         else
#             echo "  WARNING: $src not found, skipping"
#         fi
#     done
# done

# echo ""
# echo "=== Renaming wheat_prod_<year>.csv -> wheat_prod_<year+1>.csv (descending) ==="
# cd "$BASE/Wheat_Production_District_wise_1998-2025"
# for year in {2024..1998}; do
#     src="wheat_prod_${year}.csv"
#     dst="wheat_prod_$((year + 1)).csv"
#     if [ -f "$src" ]; then
#         mv -v "$src" "$dst"
#     else
#         echo "  WARNING: $src not found, skipping"
#     fi
# done

# echo ""
# echo "Done."

