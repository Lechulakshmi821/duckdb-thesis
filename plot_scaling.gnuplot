set terminal pngcairo size 900,500
set output "forward_ad_scaling.png"

set datafile separator ","
set title "Forward-Mode AD Runtime Scaling"
set xlabel "Number of tuples"
set ylabel "Runtime (seconds)"
set grid
set key off

plot "forward_ad_scaling.csv" using 1:3 every ::1 with points pt 7 ps 1.5
