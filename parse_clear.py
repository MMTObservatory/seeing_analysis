from pathlib import Path
from datetime import datetime
import pytz

with open(Path(__file__).parent / "data" / "clearing.txt", "r") as file:
    content = file.read()

cleared = {}
cleared_times = []

for line in content.splitlines():
    if "Clearing" in line:
        time, _, _, message = line.split(" - ")
        dt = datetime.strptime(time, "%Y-%m-%d %H:%M:%S,%f")
        # Arizona does not observe daylight saving time, so use 'America/Phoenix'
        arizona = pytz.timezone('America/Phoenix')
        dt_local = arizona.localize(dt)
        dt_utc = dt_local.astimezone(pytz.utc)
        date_str = dt_utc.strftime("%Y-%m-%d")
        if date_str not in cleared:
            cleared[date_str] = []

        cleared[date_str].append(dt_utc)
        cleared_times.append(dt_utc)

print(len(cleared_times), "clearing times found.")

for date, times in cleared.items():
    print(f"{date}: {len(times)} clearing times")
    for time in times:
        print(f"  {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")  # Print in UTC with timezone info
