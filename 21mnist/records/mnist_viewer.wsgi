#!/usr/bin/python3
# add the following in apache config file (e.g., /etc/apache2/sites-available/default-ssl.conf)
#WSGIDaemonProcess mnist user=share group=share home=/home/share/public_html/parallel-distributed/21mnist/records
#WSGIScriptAlias /mnist_viewer /home/share/public_html/parallel-distributed/21mnist/records/mnist_viewer.wsgi
#WSGIProcessGroup mnist
#WSGIApplicationGroup %{GLOBAL}

from mnist_viewer import application
