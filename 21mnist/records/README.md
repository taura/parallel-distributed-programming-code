
* 
```
submit < mnist.log
```
will write to `mnist_records/a.sqlite`

* https://taulec.zapto.org/mnist_viewer

consults `/etc/apache2/sites-enabled/default-ssl.conf`

and based on

```
WSGIDaemonProcess mnist user=share group=share home=/home/share/public_html/parallel-distributed/21mnist/records
WSGIScriptAlias /mnist_viewer /home/share/public_html/parallel-distributed/21mnist/records/mnist_viewer.wsgi
WSGIProcessGroup mnist
WSGIApplicationGroup %{GLOBAL}
```

it will invoke

```
/home/share/public_html/parallel-distributed/21mnist/records/mnist_viewer.wsgi
```

which then invokes

```
/home/share/public_html/parallel-distributed/21mnist/records/mnist_viewer.py
```

which then reads

```
/home/share/public_html/parallel-distributed/21mnist/records/mnist_records/a.sqlite
```

