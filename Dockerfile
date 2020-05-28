FROM pklehre/niso2020-lab2-msc

ADD beer_sales /bin
ADD jtm812.py /bin

CMD ["-username" ,"jtm812","-submission", "python jtm812.py"]


