n_imgs := 10000

url := http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
targz := $(notdir $(url))
bin_dir := cifar-10-batches-bin
bin := $(bin_dir)/data_batch_1.bin
img_dir := imgs
img_prefix := $(img_dir)/i

img_idxs := $(shell seq -w 0 $(shell echo $$(($(n_imgs) - 1))))

ppm := $(foreach i,$(img_idxs),$(img_prefix)$(i).ppm)
png := $(patsubst %.ppm,%.png,$(ppm))

index := $(img_dir)/index.html

all : $(index)

dl : $(bin)

$(targz) :
	wget -O $@ $(url)
	touch $@

$(bin) : $(targz)
	tar -x -f $<
	touch $@

../vgg.g++ : 
	cd .. && $(MAKE)

$(img_dir)/created : ../vgg.g++ $(bin)
	mkdir -p $@
	../vgg.g++ -d $(bin) -D $(img_prefix) -m 0
# --partial_data $(n_imgs)

$(ppm) : %.ppm : $(img_dir)/created
	ls $@
	touch $@

$(png) : %.png : %.ppm
	convert $< $@

$(index) : % : $(png)
#	for p in $(png); do echo "<img src=\"$$(basename $${p})\" />" ; done > $@

.DELETE_ON_ERROR:
