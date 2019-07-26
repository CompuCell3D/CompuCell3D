from collections import defaultdict


class WindowInventory(object):
    def __init__(self):

        self.inventory_dict = defaultdict(lambda: defaultdict(int))
        self.inventory_counter = 0

    def get_counter(self):

        return self.inventory_counter

    def set_counter(self, counter):

        self.inventory_counter = counter

    def add_to_inventory(self, obj, obj_type):

        self.inventory_dict[obj_type][self.inventory_counter] = obj
        self.inventory_counter += 1

    def getWindowsItems(self, win_type):

        for winId, win in self.inventory_dict[win_type].items():
            yield winId, win

    def remove_from_inventory(self, obj):

        winTypeToRemove = None
        winIdToRemove = None

        for winType, winDict in self.inventory_dict.items():
            for winId, win in winDict.items():

                if win == obj:
                    winIdToRemove = winId
                    winTypeToRemove = winType

                    break

        if winIdToRemove is not None and winTypeToRemove is not None:

            try:
                del self.inventory_dict[winTypeToRemove][winIdToRemove]
            except KeyError:
                pass

        print('AFTER REMOVING WINDOW self.inventory_dict = ', self.inventory_dict)

    def values(self):
        win_list = [val for dict_val in list(self.inventory_dict.values()) for val in list(dict_val.values())]

        return win_list

    def __str__(self):
        return self.inventory_dict.__str__()
