set terminal pngcairo size 900,560 enhanced font 'Verdana,11'
set output 'ad_scaling.png'
set title "AD Runtime Scaling in DuckDB (Release Build)" font 'Verdana,13'
set xlabel "Number of tuples"
set ylabel "Median runtime (seconds)"
set grid
set key top left
set datafile separator ","
set yrange [0:*]
plot 'ad_comparison_median_release.csv' using 1:2 with linespoints lw 2 pt 7 ps 1.3 lc rgb '#7B3FB8' title 'Baseline', \
     ''                                 using 1:3 with linespoints lw 2 pt 5 ps 1.3 lc rgb '#2E8B57' title 'Forward AD', \
     ''                                 using 1:4 with linespoints lw 2 pt 9 ps 1.3 lc rgb '#5BC8E8' title 'Reverse AD'
