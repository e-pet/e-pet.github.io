---
title: Beautiful boxplots in pgfplots
permalink: /posts/2018/2018-03-21-pgfplots-boxplot/
date: 2018-03-21
read_time: False
tags:
  - Visualization
  - Publishing
  - Latex
---
Recently, I wanted to create boxplots from a data file for a paper I was writing using [pgfplots](http://pgfplots.sourceforge.net/). Turns out that's more difficult than expected, especially since the (otherwise very useful) documentation is a bit meager in this point.

To save others the hassle, here's the result of my efforts ([link to PDF version](/images/2018-03-21-pgfplots-boxplot.pdf)):

<figure style="width: 450px" class="align-center">
  <a href="/images/2018-03-21-pgfplots-boxplot.png" title="Pgfplots Boxplot Example" alt="Pgfplots Boxplot Example">
  <img src="/images/2018-03-21-pgfplots-boxplot.png"></a>
</figure>

Some "features" of this plot that took me a while to set up correctly:
* Box design with fill color and black border
* Nice color scheme (based on [ColorBrewer](http://colorbrewer2.org/#))
* Clean plot without any unnecessary clutter
* Getting pgfplots to work with data in row format (instead of column format, as expected by the boxplot  function)

This is the code I used to generate the figure ([link to code file](https://github.com/e-pet/e-pet.github.io/blob/master/snippets/2018-03-21-pgfplots-boxplot.tex):
``` TeX
\documentclass{standalone}
\usepackage{pgfplots}
% Nice color sets, see see http://colorbrewer2.org/	
\usepgfplotslibrary{colorbrewer}
% initialize Set1-4 from colorbrewer (we're comparing 4 classes),
\pgfplotsset{compat = 1.15, cycle list/Set1-8} 
% Tikz is loaded automatically by pgfplots
\usetikzlibrary{pgfplots.statistics, pgfplots.colorbrewer} 
% provides \pgfplotstabletranspose
\usepackage{pgfplotstable}
\usepackage{filecontents}

\begin{filecontents*}{data.csv}
22, 26, 30, 17, 45
10, 15, 13, 12, 17
12, 30, 6,  57, 10
33, 38, 36, 25, 24
\end{filecontents*}

\begin{document}
\begin{tikzpicture}
	\pgfplotstableread[col sep=comma]{data.csv}\csvdata
	% Boxplot groups columns, but we want rows
	\pgfplotstabletranspose\datatransposed{\csvdata} 
	\begin{axis}[
		boxplot/draw direction = y,
		x axis line style = {opacity=0},
		axis x line* = bottom,
		axis y line = left,
		enlarge y limits,
		ymajorgrids,
		xtick = {1, 2, 3, 4},
		xticklabel style = {align=center, font=\small, rotate=60},
		xticklabels = {Apples, Oranges, Bananas, Melons},
		xtick style = {draw=none}, % Hide tick line
		ylabel = {Juiciness},
		ytick = {20, 40}
	]
		\foreach \n in {1,...,4} {
			\addplot+[boxplot, fill, draw=black] table[y index=\n] {\datatransposed};
		}
	\end{axis}
\end{tikzpicture}
\end{document}
```

------
