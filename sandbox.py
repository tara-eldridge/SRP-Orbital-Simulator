import datetime

date_1 = '2024-01-02'

print (f"date_1 = {date_1}")

day = date_1[len(date_1) - 2:len(date_1)]
print (f"day = {day}")

day2 = str(int(date_1[len(date_1) - 2:len(date_1)]) + 1).zfill(2)

print (f"day2 = {day2}")

date_2 = date_1[0:len(date_1) - 2] + day2

print (f"date_2 = {date_2}")

start_date = '2024-02-01'

date_1 = datetime.datetime.strptime(start_date, '%Y-%m-%d')

end_date = date_1 + datetime.timedelta(days=28)

print (f"end_date = {str(end_date)[0:10]}")

print ('---------------------------------')

def add_day(start_date):
    
    import calendar as cal
    start_day = start_date[len(start_date) - 2:len(start_date)]
    start_month = start_date[5:7]
    start_year = start_date[0:4]

    print (f"start_day = {start_day}")
    print (f"start_month = {start_month}")
    print (f"start_year = {start_year}")

    print (f"cal.isleap(int(start_year)) = {cal.isleap(int(start_year))}")

    day2 = str(int(start_date[len(start_date) - 2:len(start_date)]) + 1).zfill(2)
    end_date = start_date[0:len(start_date) - 2] + day2
    return end_date

start_date = '2024-01-02'
end_date = add_day(start_date)


# This class carries the major bodies physical parameters
class major_bodies:
  def __init__(self):
    self.au = 149597870.7 # km
    self.day = 86400 # seconds
    self.mu = 1.32712440018e11 # km^3/s^2
    self.Re = 6378.14 # km