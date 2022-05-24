class Color:
    def __init__(self, _in=None):
        if _in is None:
            self.r = 0
            self.g = 0
            self.b = 0
        elif isinstance(_in, str):
            c = self.from_str_rgb(_in)
            self.r, self.g, self.b = c.r, c.g, c.b
        elif isinstance(_in, tuple):
            self.r, self.g, self.b = _in

        self.a = 255

    def __str__(self):
        return "#" + "".join([hex(x)[2:].zfill(2) for x in self.to_tuple()])

    def red(self):
        return self.r

    def green(self):
        return self.g

    def blue(self):
        return self.b

    def alpha(self):
        return self.a

    @staticmethod
    def from_tuple_rgb(_tuple: tuple):
        c = Color()
        c.r, c.g, c.b = _tuple
        return c

    def to_tuple(self) -> tuple:
        return self.r, self.g, self.b

    @staticmethod
    def from_str_rgb(_str: str):
        c = Color()
        c.r, c.g, c.b = int(_str[1:3], 16), int(_str[3:5], 16), int(_str[5:7], 16)
        return c

    def to_str_rgb(self) -> str:
        return self.__str__()

    def to_json(self) -> str:
        return self.to_str_rgb()


class Point2D:
    def __init__(self, x: int = 0, y: int = 0):
        self.x = x
        self.y = y

    def __str__(self):
        return str(self.x) + ',' + str(self.y)


class Size2D:
    def __init__(self, width: int = 0, height: int = 0):
        self.width = width
        self.height = height

    def __str__(self):
        return str(self.width) + ',' + str(self.height)
