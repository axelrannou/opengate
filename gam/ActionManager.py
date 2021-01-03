import gam
import gam_g4 as g4


class ActionManager(g4.G4VUserActionInitialization):
    """
    TODO
    """

    def __init__(self, source):
        g4.G4VUserActionInitialization.__init__(self)
        # list of G4 Master source
        self.g4_PrimaryGenerator = []
        self.g4_main_PrimaryGenerator = None
        self.source_manager = source
        self.g4_RunAction = []
        self.g4_EventAction = []
        self.g4_TrackingAction = []

    def __del__(self):
        print('ActionManager destructor')
        print(self.g4_PrimaryGenerator)
        print(self.g4_main_PrimaryGenerator)
        print(self.source_manager)
        print(self.g4_RunAction)
        print(self.g4_EventAction)
        print(self.g4_TrackingAction)

    def BuildForMaster(self):
        # This function is call only in MT mode, for the master thread
        if not self.g4_main_PrimaryGenerator:
            self.g4_main_PrimaryGenerator = self.source_manager.build()

    def Build(self):
        # In MT mode the same method is invoked
        # for each worker thread, so all user action classes
        # are defined thread-locally.

        # If MT is not enabled, need to create the main source
        if not self.g4_main_PrimaryGenerator:
            p = self.g4_main_PrimaryGenerator = self.source_manager.build()
        else:
            # else create a source for each thread
            p = self.source_manager.create_g4_source_manager()

        #return  ## FIXME -> setUserAction lead to seg fault

        self.SetUserAction(p)
        self.g4_PrimaryGenerator.append(p)

        # set the actions for Run
        #ra = gam.RunAction()  # FIXME why not on cpp side ?
        ra = g4.GamRunAction()  # FIXME
        self.SetUserAction(ra)
        self.g4_RunAction.append(ra)

        # set the actions for Event
        ea = g4.GamEventAction()
        self.SetUserAction(ea)
        self.g4_EventAction.append(ea)

        # set the actions for Track
        ta = g4.GamTrackingAction()
        self.SetUserAction(ta)
        self.g4_TrackingAction.append(ta)
