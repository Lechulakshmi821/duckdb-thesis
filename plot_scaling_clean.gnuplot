set terminal pngcairo size 1000,600 enhanced font "Arial,14"
set output "forward_ad_scaling_clean.png"

set datafile separator ","
set title "Forward-Mode AD Runtime Scaling"
set xlabel "Number of tuples"
set ylabel "Mean runtime (seconds)"

set grid ytics xtics
set key left top

set style line 1 lt 1 lw 3 pt 7 ps 1.5

plot "forward_ad_scaling_mean.csv" using 1:2 every ::1 with linespoints linestyle 1 title "Forward-mode AD"
