from cc3d.cpp import CompuCell


class CellInventoryWatcher(CompuCell.CellInventoryWatcherDir):
    """
    Interface for handling events associated with changes to the cell inventory
    """

    def __init__(self, cell_inventory: CompuCell.CellInventory):
        super().__init__()
        CompuCell.makeCellInventoryWatcher(self, cell_inventory)

    def onCellAdd(self, cell: CompuCell.CellG):
        return self.on_cell_add(cell)

    def onCellRemove(self, cell: CompuCell.CellG):
        return self.on_cell_remove(cell)

    def on_cell_add(self, cell: CompuCell.CellG):
        """
        Callback for handling when a cell is added to the cell inventory

        :param cell: added cell
        :return: None
        """

        pass

    def on_cell_remove(self, cell: CompuCell.CellG):
        """
        Callback for handling when a cell is removed from the cell inventory

        :param cell: removed cell
        :return: None
        """

        pass
