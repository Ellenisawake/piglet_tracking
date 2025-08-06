from datetime import datetime
import os
import numpy as np
import json


def log_no_print(writer, text):
    writer.write('%s\n' % text)
    writer.flush()


def log_with_print(writer, text):
    print(text)
    writer.write('%s\n' % text)
    writer.flush()


def get_date_to_save():
    today = str(datetime.date(datetime.now()))
    year_month_date = today.split('-')
    date_to_save = year_month_date[0][2:] + year_month_date[1] + year_month_date[2]
    return date_to_save


def get_time_to_print():
    time = str(datetime.time(datetime.now()))
    time = time[:8]
    return time
