from datetime import date
today=date.today()
dateStr=str(today.year)+str(today.month).zfill(2)+str(today.day).zfill(2)
print dateStr

