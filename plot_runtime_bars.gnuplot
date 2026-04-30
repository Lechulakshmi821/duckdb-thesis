set terminal pngcairo size 1200,700 enhanced font "Arial,14"
set output "runtime_bars.png"
set title "Runtime Comparison of AD Methods"
set xlabel "Number of tuples"
set ylabel "Median Runtime (seconds)"
set grid ytics
set key outside top center
set style data histograms
set style histogram clustered gap 1
set style fill solid 0.8 border -1
set boxwidth 0.8
plot \
"baseline.dat" using 2:xtic(1) title "baseline", \
"forward.dat" using 2 title "Forward AD", \
"reverse.dat" using 2 title "Reverse AD"
