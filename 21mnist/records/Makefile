
user=$(shell whoami)
group=$(shell id -n -g $(user))

/usr/local/bin/submit : setuid_submit
	sudo cp $< $@
	sudo chown $(user):$(group) $@
	sudo chmod u+s $@

setuid_submit : setuid_submit.c
	gcc -Wall -Wextra -DPROG=\"$(shell pwd)/submit.py\" -o $@ $<

