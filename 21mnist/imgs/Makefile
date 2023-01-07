
dirs := $(wildcard t*/00000_00500)
htmls := $(patsubst %,%/index.html,$(dirs))

pgms := $(foreach dir,$(dirs),$(wildcard $(dir)/*.pgm))
pngs := $(patsubst %.pgm,%.png,$(pgms))

targets := $(htmls)

all : $(targets)

$(pngs) : %.png : %.pgm
	convert $< $@

$(htmls) : %/index.html : $(pngs)
	./make_index_html.py $*
