import numpy as np
from dateutil import parser, tz

import rsciio.utils._date_time as dtt
from rsciio.utils._dictionary import DTBox


def _get_example(date, time, time_zone=None):
    md = DTBox({"General": {"date": date, "time": time}}, box_dots=True)
    if time_zone:
        md.set_item("General.time_zone", time_zone)
        dt = parser.parse("%sT%s" % (date, time))
        dt = dt.replace(tzinfo=tz.gettz(time_zone))
        iso = dt.isoformat()
    else:
        iso = "%sT%s" % (date, time)
        dt = parser.parse(iso)
    return md, dt, iso


md1, dt1, iso1 = _get_example("2014-12-27", "00:00:00", "UTC")
serial1 = 42000.00

md2, dt2, iso2 = _get_example("2124-03-25", "10:04:48", "EST")
serial2 = 81900.62833333334

md3, dt3, iso3 = _get_example("2016-07-12", "22:57:32")
serial3 = 42563.95662037037


def test_serial_date_to_ISO_format():
    iso_1 = dtt.serial_date_to_ISO_format(serial1)
    dt1_local = dt1.astimezone(tz.tzlocal())
    assert iso_1[0] == dt1_local.date().isoformat()
    assert iso_1[1] == dt1_local.time().isoformat()
    assert iso_1[2] == dt1_local.tzname()

    iso_2 = dtt.serial_date_to_ISO_format(serial2)
    dt2_local = dt2.astimezone(tz.tzlocal())
    assert iso_2[0] == dt2_local.date().isoformat()
    # The below line will/can fail due to accuracy loss when converting to serial date:
    # We therefore truncate milli/micro seconds
    assert iso_2[1][:8] == dt2_local.time().isoformat()
    assert iso_2[2] == dt2_local.tzname()

    iso_3 = dtt.serial_date_to_ISO_format(serial3)
    dt3_aware = dt3.replace(tzinfo=tz.tzutc())
    dt3_local = dt3_aware.astimezone(tz.tzlocal())
    assert iso_3[0] == dt3_local.date().isoformat()
    assert iso_3[1] == dt3_local.time().isoformat()
    assert iso_3[2] == dt3_local.tzname()


def test_ISO_format_to_serial_date():
    res1 = dtt.ISO_format_to_serial_date(
        dt1.date().isoformat(), dt1.time().isoformat(), timezone=dt1.tzname()
    )
    np.testing.assert_allclose(res1, serial1, atol=1e-5)
    dt = dt2.astimezone(tz.tzlocal())
    res2 = dtt.ISO_format_to_serial_date(
        dt.date().isoformat(), dt.time().isoformat(), timezone=dt.tzname()
    )
    np.testing.assert_allclose(res2, serial2, atol=1e-5)
    res3 = dtt.ISO_format_to_serial_date(
        dt3.date().isoformat(), dt3.time().isoformat(), timezone=dt3.tzname()
    )
    np.testing.assert_allclose(res3, serial3, atol=1e-5)


def test_datetime_to_serial_date():
    np.testing.assert_allclose(dtt.datetime_to_serial_date(dt1), serial1, atol=1e-5)
    np.testing.assert_allclose(dtt.datetime_to_serial_date(dt2), serial2, atol=1e-5)
    np.testing.assert_allclose(dtt.datetime_to_serial_date(dt3), serial3, atol=1e-5)


def _get_example(date, time, time_zone=None):
    md = {"General": {"date": date, "time": time}}
    if time_zone:
        md["General"]["time_zone"] = time_zone
        dt = parser.parse(f"{date}T{time}")
        dt = dt.replace(tzinfo=tz.gettz(time_zone))
        iso = dt.isoformat()
    else:
        iso = f"{date}T{time}"
        dt = parser.parse(iso)
    return md, dt, iso


md1, dt1, iso1 = _get_example("2014-12-27", "00:00:00", "UTC")
serial1 = 42000.00

md2, dt2, iso2 = _get_example("2124-03-25", "10:04:48", "EST")
serial2 = 81900.62833333334

md3, dt3, iso3 = _get_example("2016-07-12", "22:57:32")
serial3 = 42563.95662037037


def test_get_date_time_from_metadata():
    assert dtt.get_date_time_from_metadata(md1) == "2014-12-27T00:00:00+00:00"
    assert (
        dtt.get_date_time_from_metadata(md1, formatting="ISO")
        == "2014-12-27T00:00:00+00:00"
    )
    assert dtt.get_date_time_from_metadata(
        md1, formatting="datetime64"
    ) == np.datetime64("2014-12-27T00:00:00.000000")
    assert dtt.get_date_time_from_metadata(md1, formatting="datetime") == dt1

    assert dtt.get_date_time_from_metadata(md2) == "2124-03-25T10:04:48-05:00"
    assert dtt.get_date_time_from_metadata(md2, formatting="datetime") == dt2
    assert dtt.get_date_time_from_metadata(
        md2, formatting="datetime64"
    ) == np.datetime64("2124-03-25T10:04:48")

    assert dtt.get_date_time_from_metadata(md3) == "2016-07-12T22:57:32"
    assert dtt.get_date_time_from_metadata(md3, formatting="datetime") == dt3
    assert dtt.get_date_time_from_metadata(
        md3, formatting="datetime64"
    ) == np.datetime64("2016-07-12T22:57:32.000000")

    assert dtt.get_date_time_from_metadata({"General": {}}) is None
    assert (
        dtt.get_date_time_from_metadata({"General": {"date": "2016-07-12"}})
        == "2016-07-12"
    )
    assert dtt.get_date_time_from_metadata({"General": {"time": "12:00"}}) == "12:00:00"
    assert (
        dtt.get_date_time_from_metadata(
            {"General": {"time": "12:00", "time_zone": "CET"}}
        )
        == "12:00:00"
    )
