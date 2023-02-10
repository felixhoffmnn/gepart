.PHONY: disposition paper debug-paper biber clean

disposition:
	@echo "Creating Disposition Paper..."
	pandoc dhbw/disposition.md --citeproc --bibliography=dhbw/bibliography.bib --csl data/citetation/acm.csl -s -o out/disposition.pdf --from markdown --template eisvogel --toc --listings --number-sections --top-level-division=chapter -V geometry:a4paper

paper: biber
	@echo "Creating Paper..."
	cd paper && pdflatex -synctex=1 -interaction=nonstopmode -output-directory out/  dokumentation.tex

debug-paper:
	@echo "Creating Paper (Debug Mode)..."
	cd paper && pdflatex -synctex=1 -output-directory out/ dokumentation.tex

biber:
	@echo "Creating Bibliography..."
	cd paper && biber dokumentation --input-directory out/ --output-directory out/

clean:
	@echo "Cleaning up (Paper and Disposition)..."
	rm -rf paper/out/*
	rm -rf dhbw/out/disposition.pdf
