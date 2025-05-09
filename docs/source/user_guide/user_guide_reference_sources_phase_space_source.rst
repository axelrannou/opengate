.. _source-phase_space-source:

Phase Space Source
=====================

Description
------------

The Phase Space source is a source type within GATE that utilizes a prebuilt Phase Space to emit particles. Each particle emitted at the beginning of an event is based on the particle’s state (position, energy, direction, particle type, and weight) stored in the Phase Space file.

To use this source, declare a `PhaseSpaceSource` within the `add_source` method. The file path of the desired Phase Space source must be provided. Only ROOT files are supported as input for this source.

By default, the parameter names corresponding to the particle states align with the ROOT output generated by the :class:`~.opengate.actors.digitizers.PhaseSpaceActor`. However, users can provide custom ROOT files and specify alternative parameter names for the particle state components. The Phase Space file is read sequentially starting from the beginning by default. The `entry_start` option allows users to specify a custom starting point within the file.

To optimize performance and reduce computational costs associated with event-by-event file access, a batch of \(N\) particles is preloaded into the computer’s RAM. The batch size \(N\) is user-definable, with 100,000 being a recommended trade-off between memory usage and performance.

Additionally, users can apply positional offsets or rotation matrices to the positions and directions read from the Phase Space file. By default, the positions and directions of particles are defined relative to the coordinates of the parent volume. Setting the `global_flag` option to `True` changes this behavior, allowing particles to be emitted according to the world coordinate system.

Below is an example Python script for defining a Phase Space source:

.. code:: python

   source = sim.add_source("PhaseSpaceSource", "phsp_source_global")
   source.attached_to = user_plane_source.name
   source.position_key = "PrePositionLocal"
   source.direction_key = "PreDirectionLocal"
   source.PDGCode_key = "PDGCode"
   source.energy_key = "KineticEnergy"
   source.weight_key = "Weight"
   source.entry_start = np.random.randint(0, 10**9, 1)[0]
   source.batch_size = 100000
   source.global_flag = False
   source.translate_position = False
   source.rotate_direction = False

This macro works in single-threaded mode. To enable multithreading, the `entry_start` parameter must be defined as a list of indices, where the length of the list corresponds to the number of allocated threads. For example:

.. code:: python

   source.entry_start = np.random.randint(0, 10**9, sim.number_of_threads)

If any of the provided `entry_start` indices exceed the size of the Phase Space file, the index will be adjusted automatically using the modulo operator relative to the file size.

Reference
---------

.. autoclass:: opengate.sources.phspsources.PhaseSpaceSource








