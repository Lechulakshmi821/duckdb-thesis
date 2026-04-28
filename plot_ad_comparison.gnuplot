set terminal pngcairo size 1000,600 enhanced font "Arial,14"
set output "ad_comparison_scaling.png"

set title "AD Runtime Scaling in DuckDB"
set xlabel "Number of tuples"
set ylabel "Median runtime (seconds)"
set grid
set key left top

set style line 1 lw 3 pt 7 ps 1.3
set style line 2 lw 3 pt 5 ps 1.3
set style line 3 lw 3 pt 9 ps 1.3

plot \
  "baseline.dat" using 1:2 with linespoints linestyle 1 title "Baseline", \
  "forward_ad.dat" using 1:2 with linespoints linestyle 2 title "Forward AD", \
  "reverse_ad.dat" using 1:2 with linespoints linestyle 3 title "Reverse AD"
