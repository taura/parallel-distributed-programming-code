

ssh you@taulec.zapto.org submit < vgg.log

will write to taulec.zapto.org:/home/tau/vgg_records/a.sqlite

https://taulec.zapto.org/viewer

consuluts /etc/apache2/sites-enabled/000-default-le-ssl.conf

and based on

WSGIScriptAlias /viewer /home/tau/public_html/lecture/parallel_distributed/parallel-distributed-handson/20vgg/records/viewer/viewer.wsgi

it will invoke

/home/tau/public_html/lecture/parallel_distributed/parallel-distributed-handson/20vgg/records/viewer/viewer.wsgi

which then

/home/tau/public_html/lecture/parallel_distributed/parallel-distributed-handson/20vgg/records/viewer/viewer.py

which then reads

/home/tau/vgg_records/a.sqlite






# note

* this directory is in a premature state
* user interface will change over time
* a quick info for those who want to use it now

# setup to see images

(can be skipped when you do not need to see classified images)

* generate data by doing
```
$ cd ../data
$ make -f data.mk
```

* move data from ../data/ by
```
$ mv ../data/imgs .
```

# run

```
$ cd ..
$ ./vgg.g++
```

# see the log

```
$ ./parse_log.py ../vgg.log
```

open index.html with your browser

you should be able to see

 * how the loss function evolved over time
 * how much time is spent on which kernel
 * history of classifications

