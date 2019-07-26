from cc3d.player5.ViewManager import SimpleViewManager


def factory(parent, ui):
    """
    Modul factory function to generate the right viewmanager type.
    
    The viewmanager is instantiated depending on the data set in
    the current preferences.
    
    @param parent parent widget (QWidget)
    @param pluginManager reference to the plugin manager object
    @return the instantiated viewmanager
    """

    """
    # They don't instantiate the object directly. They use plugin manager!
    
    vm = pluginManager.getPluginObject("viewmanager", viewManagerStr)
    if vm is None:
        # load tabview view manager as default
        vm = pluginManager.getPluginObject("viewmanager", "tabview")
        if vm is None:
            raise RuntimeError("Could not create a viemanager object.")
    """
    vm = SimpleViewManager.SimpleViewManager(ui)

    return vm
