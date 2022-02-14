import gam_gate as gam
import gam_g4 as g4
import numpy as np
import itk
from scipy.spatial.transform import Rotation


class HitsProjectionActor(g4.GamHitsProjectionActor, gam.ActorBase):
    """
    FIXME TODO
    """

    type_name = 'HitsProjectionActor'

    @staticmethod
    def set_default_user_info(user_info):
        gam.ActorBase.set_default_user_info(user_info)
        # fixme add options here
        mm = gam.g4_units('mm')
        user_info.output = 'projections.mhd'
        user_info.input_hits_collections = ['Hits']
        user_info.spacing = [4 * mm, 4 * mm]
        user_info.dimension = [128, 128]

    def __init__(self, user_info):
        gam.ActorBase.__init__(self, user_info)
        g4.GamHitsProjectionActor.__init__(self, user_info.__dict__)
        actions = {'StartSimulationAction', 'BeginOfRunAction', 'EndSimulationAction'}
        self.AddActions(actions)
        self.image = None
        if len(user_info.input_hits_collections) < 1:
            gam.fatal(f'Error, not input hits collection.')

    def __del__(self):
        pass

    def __str__(self):
        s = f'HitsProjectionActor {self.user_info.name}'
        return s

    def StartSimulationAction(self):  # not needed, only if need to do something in python
        # position according to the mother volume
        vol = self.simulation.volume_manager.get_volume(self.user_info.mother)
        solid = vol.g4_physical_volumes[0].GetLogicalVolume().GetSolid()
        pMin = g4.G4ThreeVector()
        pMax = g4.G4ThreeVector()
        solid.BoundingLimits(pMin, pMax)
        # check size and spacing
        if len(self.user_info.dimension) != 2:
            gam.fatal(f'Error, the dimension must be 2D while it is {self.user_info.dimension}')
        if len(self.user_info.spacing) != 2:
            gam.fatal(f'Error, the spacing must be 2D while it is {self.user_info.spacing}')
        self.user_info.dimension.append(1)
        self.user_info.spacing.append(1)
        # define the new size and spacing according to the nb of channels and volume shape
        size = np.array(self.user_info.dimension)
        spacing = np.array(self.user_info.spacing)
        # real_size = size.copy()
        # channel_size = size.copy()
        # real_size[2] = len(self.user_info.input_hits_collections) * len(self.simulation.run_timing_intervals)
        size[2] = len(self.user_info.input_hits_collections) * len(self.simulation.run_timing_intervals)
        # channel_size[2] = len(self.user_info.input_hits_collections)
        spacing[2] = (pMax[2] - pMin[2]) / size[2]
        # print(size, real_size, channel_size)
        print(size, spacing)
        # create image
        self.image = gam.create_3d_image(size, spacing)

        '''self.img_center = -size * spacing / 2.0 + spacing / 2.0
        # define the global transformation of the volume
        vol = vol.g4_physical_volumes[0].GetName()
        translation, rotation = gam.get_transform_world_to_local(vol)
        print('initial transfo', translation, rotation)
        t = gam.get_translation_from_rotation_with_center(Rotation.from_matrix(rotation), self.img_center)
        # compute the corresponding origin of the image
        origin = translation + self.img_center - t
        self.image.SetOrigin(origin)
        self.image.SetDirection(rotation)
        print('Start origin', origin)
        print('Start rot', rotation)'''
        print()
        print('START SIMULATION PROJECTION')
        gam.attach_image_to_volume(self.simulation, self.image, self.user_info.mother)
        # update the cpp image and Start
        gam.update_image_py_to_cpp(self.image, self.fImage, True)
        g4.GamHitsProjectionActor.StartSimulationAction(self)

    def BeginOfRunAction(self, run):
        print('I am in run', run.GetRunID())
        print(self.user_info.mother)
        '''vol = self.simulation.volume_manager.get_volume(self.user_info.mother)
        vol = vol.g4_physical_volumes[0].GetName()
        translation, rotation = gam.get_transform_world_to_local(vol)
        print('run transfo', translation, rotation)
        t = gam.get_translation_from_rotation_with_center(Rotation.from_matrix(rotation), self.img_center)
        origin = translation + self.img_center - t
        '''
        gam.attach_image_to_volume(self.simulation, self.image, self.user_info.mother)
        gam.update_image_py_to_cpp(self.image, self.fImage, False)
        super().BeginOfRunAction(run)

    def EndSimulationAction(self):
        g4.GamHitsProjectionActor.EndSimulationAction(self)
        self.image = gam.get_cpp_image(self.fImage)
        print(self.image.GetOrigin())
        print(self.image.GetDirection())
        itk.imwrite(self.image, gam.check_filename_type(self.user_info.output))
