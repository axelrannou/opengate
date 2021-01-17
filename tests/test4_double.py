#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gam

# verbose level
gam.log.setLevel(gam.INFO)

# create the simulation
sim = gam.Simulation()

# main options
sim.set_g4_verbose(False)
sim.set_g4_visualisation_flag(True)
sim.set_g4_multi_thread(False)
sim.set_g4_random_engine("MersenneTwister", 123654)

# set the world size like in the Gate macro
m = gam.g4_units('m')
world = sim.get_volume_info('World')
world.size = [3 * m, 3 * m, 3 * m]

# add a simple waterbox volume
waterbox = sim.add_volume('Box', 'Waterbox')
cm = gam.g4_units('cm')
waterbox.size = [40 * cm, 40 * cm, 40 * cm]
waterbox.translation = [0 * cm, 0 * cm, 25 * cm]
waterbox.material = 'G4_WATER'

# physic list # FIXME will be changed
# print('Phys lists :', sim.get_available_physicLists())

# default source for tests
keV = gam.g4_units('keV')
mm = gam.g4_units('mm')
Bq = gam.g4_units('Bq')
source = sim.add_source('Generic', 'Default')
source.particle = 'gamma'
source.energy.mono = 80 * keV
source.direction.type = 'momentum'
source.direction.momentum = [0, 0, 1]
source.activity = 200000 * Bq

# add stat actor
sim.add_actor('SimulationStatisticsActor', 'Stats')

# create G4 objects
sim.initialize()

# start simulation
#sim.apply_g4_command("/run/verbose 1")
gam.source_log.setLevel(gam.RUN)
sim.start()

stats = sim.get_actor('Stats')
print(stats)

## redo it

sim.initialize()
sim.start()
stats = sim.get_actor('Stats')
print(stats)
