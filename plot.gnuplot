set terminal pngcairo size 900,450
set output "lr_bars_50k_threads1.png"

set datafile separator ","
set style data histograms
set style histogram clustered gap 1
set style fill solid 1.0 border -1
set boxwidth 0.6

set title "Linear Regression GD (50K tuples, 20 iterations, threads=1)"
set ylabel "Runtime (seconds)"
set grid ytics

plot "times.csv" using 2:xtic(1) title ""
