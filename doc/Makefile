# Technical companion document.
#   make            build main.pdf with latexmk
#   make figures    copy the figures the document embeds from ../results
#   make clean      remove build artefacts (keeps main.pdf)
#
# No local LaTeX?  Upload this folder (main.tex + figures/) to Overleaf as-is.

RES = ../results
FIG = figures

FIGS = \
	$(FIG)/comparison_uffizi.png \
	$(FIG)/comparison_gray.png \
	$(FIG)/comparison_d65.png \
	$(FIG)/comparison_fl1.png \
	$(FIG)/basis.png \
	$(FIG)/coeffs.png \
	$(FIG)/split_sum.png

.PHONY: all figures clean

all: main.pdf

main.pdf: main.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

figures: $(FIGS)

$(FIG)/comparison_uffizi.png: $(RES)/comparison.png
	cp $< $@
$(FIG)/comparison_gray.png:   $(RES)/comparison_gray.png
	cp $< $@
$(FIG)/comparison_d65.png:    $(RES)/comparison_d65.png
	cp $< $@
$(FIG)/comparison_fl1.png:    $(RES)/comparison_fl1.png
	cp $< $@
$(FIG)/basis.png:             $(RES)/basis_plot.png
	cp $< $@
$(FIG)/coeffs.png:            $(RES)/coeffs_plot.png
	cp $< $@
$(FIG)/split_sum.png:         $(RES)/split_sum_plot.png
	cp $< $@

clean:
	latexmk -c
