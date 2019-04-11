import datetime

import numpy as np


def time_handle(time_in_seconds, start_time=datetime.time(7, 0, 0)):
    time_in_seconds = time_in_seconds % 86400
    temp = datetime.timedelta(seconds=time_in_seconds)

    # if temp.days>0:
    #     temp = temp - datetime.timedelta(days=1)

    tsim = np.array(list(map(int, str(temp).split(":"))))
    tstart = np.array([start_time.hour, start_time.minute, start_time.second])

    tnew = tsim + tstart

    if tnew[2] > 60:
        quotient = tnew[2] // 60
        remainder = tnew[2] % 60

        tnew[2] = remainder
        tnew[1] = tnew[1] + quotient

    if tnew[1] > 60:
        quotient = tnew[1] // 60
        remainder = tnew[1] % 60

        tnew[1] = remainder
        tnew[0] = (tnew[0] + quotient)

    if tnew[0] >= 24:
        tnew[0] = tnew[0] % 24

    return datetime.time(tnew[0], tnew[1], tnew[2])
