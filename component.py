class Component:
    def __init__(self, left_most, top_most, box_width, box_height, co_area):
        self.left_most = left_most
        self.top_most = top_most
        self.box_width = box_width
        self.box_height = box_height
        self.co_area = co_area

    def __str__(self):
        return f'(Component:\nleft_most={self.left_most}, top_most={self.top_most}, box_width={self.box_width}, box_height={self.box_height}, co_area={self.co_area})\n\n'

    def __repr__(self):
        return str(self)

    def get_right_most(self):
        return self.left_most + self.box_width

    def get_bottom_most(self):
        return self.top_most + self.box_height