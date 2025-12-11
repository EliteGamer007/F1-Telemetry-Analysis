import fastf1
import fastf1.plotting
import matplotlib.pyplot as plt
import numpy as np

fastf1.Cache.enable_cache('cache') 
fastf1.plotting.setup_mpl(misc_mpl_mods=False)


print("Loading session data...")
session = fastf1.get_session(2025, 'Abu Dhabi', 'R')
session.load()

laps_nor = session.laps.pick_drivers('NOR')
laps_pia = session.laps.pick_drivers('PIA')

fastest_nor = laps_nor.pick_fastest()
fastest_pia = laps_pia.pick_fastest()

tel_nor = fastest_nor.get_telemetry().add_distance()
tel_pia = fastest_pia.get_telemetry().add_distance()

color_nor = fastf1.plotting.get_driver_color('NOR', session=session)
color_pia = fastf1.plotting.get_driver_color('PIA', session=session)

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(tel_nor['Distance'], tel_nor['Speed'], 
        color=color_nor, 
        label='Norris')

ax.plot(tel_pia['Distance'], tel_pia['Speed'], 
        color=color_pia, 
        linestyle='--',
        label='Piastri')

ax.set_xlabel('Distance (m)')
ax.set_ylabel('Speed (km/h)')
ax.set_title('Fastest Lap Comparison: Abu Dhabi 2025 (NOR vs PIA)')
ax.legend()
plt.show()

