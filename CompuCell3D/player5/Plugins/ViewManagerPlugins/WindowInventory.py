from collections import defaultdict


class WindowInventory(object):
    def __init__(self):

        self.inventory_dict = defaultdict(lambda: defaultdict(int))

        # self.inventory_dict = {}
        self.inventory_counter = 0
    #     self.screenshot_win = None
    #
    # def set_screenshot_win(self, win):
    #
    #     self.screenshot_win =  win
    #
    # def get_screenshot_win(self):
    #
    #     return self.screenshot_win
    #
    # def remove_screenshot_win(self):
    #     self.screenshot_win = None

    def get_counter(self):

        return self.inventory_counter

    def set_counter(self, counter):

        self.inventory_counter = counter


    def add_to_inventory(self,  obj, obj_type):

        self.inventory_dict[obj_type][self.inventory_counter] = obj
        self.inventory_counter += 1
        # print 'self.inventory_dict=',self.inventory_dict

        # self.inventory_dict[self.inventory_counter] = obj
        # self.inventory_counter += 1

    # def remove_from_inventory_by_name(self, obj_name):
    #     try:
    #         del self.inventory_dict[obj_name]
    #         self.inventory_counter -= 1
    #     except KeyError:
    #         pass
    def getWindowsItems(self, win_type):

        for winId, win in self.inventory_dict[win_type].iteritems():
            yield winId, win

    def remove_from_inventory(self, obj):



        winTypeToRemove = None
        winIdToRemove = None

        for winType, winDict in self.inventory_dict.iteritems():
            for winId, win in winDict.iteritems():

                if win == obj:
                    winIdToRemove = winId
                    winTypeToRemove = winType

                    break

        if winIdToRemove is not None and winTypeToRemove is not None:

            try:
                del self.inventory_dict[winTypeToRemove][winIdToRemove]
            except KeyError:
                pass

        print 'AFTER REMOVING WINDOW self.inventory_dict = ', self.inventory_dict
        # import sys
        # sys.exit()
        # obj_name_to_remove = None
        # for key, val in self.inventory_dict.iteritems():
        #
        #     if val == obj:
        #
        #         obj_name_to_remove = key
        #         break
        #
        # if obj_name_to_remove is not None:
        #
        #     try:
        #         del self.inventory_dict[obj_name_to_remove]
        #     except KeyError:
        #         pass



    # def values(self):return self.inventory_dict.values()
    def values(self):
        win_list = [val for dict_val in self.inventory_dict.values() for val in dict_val.values()]
        # if self.screenshot_win:
        #     win_list.append(self.screenshot_win)
        return win_list

    def __str__(self):
        return self.inventory_dict.__str__()
