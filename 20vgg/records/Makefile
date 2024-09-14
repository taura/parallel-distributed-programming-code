
/usr/local/bin/submit : setuid_submit
	sudo cp $< $@
	sudo chown tau:tau $@
	sudo chmod u+s $@

setuid_submit : setuid_submit.c
	gcc -Wall -Wextra -DPROG=\"$(shell pwd)/submit.py\" -o $@ $<

