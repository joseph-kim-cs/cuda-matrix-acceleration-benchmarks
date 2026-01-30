gcc matrix-cpu.c -O2 -o matrix-cpu
# command to compile the CPU matrix multiplication program

N_VALUES=(256 512 1024 1536 2048)

OUTPUT_DIR="../data"
OUTPUT_FILE="$OUTPUT_DIR/cpu-results.csv"

mkdir -p $OUTPUT_DIR

# CSV header
echo "N,time_seconds" > $OUTPUT_FILE

echo "Running CPU matrix multiplication benchmarks..."

for N in "${N_VALUES[@]}"; do
    echo "Running N=$N"
    OUTPUT=$(./matrix-cpu $N)
    TIME=$(echo $OUTPUT | awk '{print $5}')
    echo "$N,$TIME" >> $OUTPUT_FILE
done

echo "Done. Results saved to $OUTPUT_FILE"
